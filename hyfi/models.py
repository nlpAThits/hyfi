#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import geoopt as gt
from hyfi import constants as cs
import hyfi.hypernn as hnn
from hyfi import utils
import geoopt.manifolds.stereographic.math as pmath

log = utils.get_logging()
POINCARE_K = torch.Tensor([-1.0]).to(cs.DEVICE)


def get_nonlin(nonlin):
    if nonlin == "tanh":
        return nn.Tanh()
    if nonlin == "relu":
        return nn.ReLU()
    if nonlin == "sigmoid":
        return nn.Sigmoid()
    return None


def define_mapping(in_metric, out_metric, c_value):
    if in_metric == out_metric:
        return lambda x: x
    elif in_metric == cs.HY and out_metric == cs.EU:
        return lambda x: pmath.logmap0(x, k=POINCARE_K)
    elif in_metric == cs.EU and out_metric == cs.HY:
        return lambda x: pmath.expmap0(x, k=POINCARE_K)
    else:
        raise ValueError(f"Wrong metrics: in_metric:'{in_metric}', out_metric:'{out_metric}'")


class MentionEncoder(nn.Module):
    """Mention encoder based on word level features, extracted with a FFNN, and a char RNN."""

    def __init__(self, char_vocab, args):
        super(MentionEncoder, self).__init__()

        self.mention_output_dim = args.space_dims * 2
        self.char_output_dim = args.space_dims
        if args.encoder_metric == cs.HY:
            self.manifold = gt.PoincareBall()
            self.word2space = hnn.MobiusLinear(args.word_emb_size, self.mention_output_dim,
                                               hyperbolic_input=True, hyperbolic_bias=True,
                                               nonlin=get_nonlin(args.men_nonlin))
            self.non_lin = lambda x: x
            self.char_rnn = hnn.MobiusRNN(args.space_dims, args.space_dims)
        else:
            self.manifold = gt.Euclidean()
            self.word2space = nn.Linear(args.word_emb_size, self.mention_output_dim)
            self.non_lin = get_nonlin(args.men_nonlin)
            self.char_rnn = hnn.EuclRNN(args.space_dims, args.space_dims)

        self.input_dropout = nn.Dropout(p=args.input_dropout)
        self.mention_attn = DistanceAttention(args, args.mention_len + 1, self.mention_output_dim)
        self.char_mapping = define_mapping(args.encoder_metric, args.attn_metric, args.c)
        self.char_midpoint = hnn.mobius_midpoint if args.attn_metric == cs.HY else hnn.euclidean_midpoint

        # char embeds
        with torch.no_grad():
            char_embeds = init_embeddings(char_vocab.size(), self.char_output_dim)
            if args.encoder_metric == cs.HY:
                char_embeds = pmath.expmap0(char_embeds, k=self.manifold.k)
        self.char_lut = gt.ManifoldParameter(char_embeds, manifold=self.manifold)

    def forward(self, mentions, mention_chars, word_lut):
        """
        :param mentions:
        :param mention_chars:
        :param word_lut:
        :return: mention_vectors: b x space_dims, chars: b x space_dims
        """
        mention_embeds = self.input_dropout(word_lut[mentions])             # batch x men_len x word_emb_size
        mention_vectors = self.word2space(mention_embeds)                   # batch * men_len x space_dim
        mention_vectors = self.non_lin(mention_vectors)

        pos_index = torch.arange(start=1, end=mentions.size(1) + 1,
                                 device=cs.DEVICE).unsqueeze(dim=0).expand(mentions.size(0), -1)
        mask = mentions > 0
        pos_index = torch.where(mask, pos_index, torch.LongTensor([0]).to(cs.DEVICE))

        mention_vectors, _ = self.mention_attn(mention_vectors, pos_index)

        char_embeds = self.char_lut[mention_chars]      # batch x char_len x char_emb_size
        char_states = self.char_rnn(char_embeds)        # batch x char_len x space_dims
        char_states = self.char_mapping(char_states)

        return mention_vectors, self.char_midpoint(char_states)


class ContextEncoder(nn.Module):

    def __init__(self, args):
        super(ContextEncoder, self).__init__()
        self.context_len = args.context_len
        self.input_dropout = nn.Dropout(p=args.input_dropout)
        self.ctx_output_dim = args.space_dims
        if args.encoder_metric == cs.HY:
            self.fwd = hnn.MobiusGRU(input_size=args.word_emb_size, hidden_size=self.ctx_output_dim,
                                     num_layers=args.num_layers,
                                     nonlin=get_nonlin(args.ctx_nonlin),
                                     hyperbolic_input=True, hyperbolic_hidden_state0=True)
            self.bkwd = hnn.MobiusGRU(input_size=args.word_emb_size, hidden_size=self.ctx_output_dim,
                                      num_layers=args.num_layers,
                                      nonlin=get_nonlin(args.ctx_nonlin),
                                      hyperbolic_input=True, hyperbolic_hidden_state0=True)
        else:
            self.fwd = hnn.EuclGRU(input_size=args.word_emb_size, hidden_size=self.ctx_output_dim,
                                   num_layers=args.num_layers,
                                   nonlin=get_nonlin(args.ctx_nonlin))
            self.bkwd = hnn.EuclGRU(input_size=args.word_emb_size, hidden_size=self.ctx_output_dim,
                                    num_layers=args.num_layers,
                                    nonlin=get_nonlin(args.ctx_nonlin))

    def forward(self, fwd_ctx, word_lut):
        """
        :param contexts: batch x seq_len
        :param lengths: batch x 1
        :returns batch x seq_len x space_dim
        """
        bkwd_ctx = fwd_ctx.flip(1)

        fwd_word_embeds = self.input_dropout(word_lut[fwd_ctx])                 # batch x seq_len x embed_dim
        bkwd_word_embeds = self.input_dropout(word_lut[bkwd_ctx])               # batch x seq_len x embed_dim

        fwd_states = self.apply_rnn(self.fwd, fwd_word_embeds)         # b x seq_len x space_dim
        bkwd_states = self.apply_rnn(self.bkwd, bkwd_word_embeds)      # b x seq_len x space_dim

        # flip backward pass to align the states to the fwd pass
        bkwd_states_flipped = bkwd_states.flip(1)

        return fwd_states, bkwd_states_flipped

    def apply_rnn(self, rnn, ctx_word_embeds):
        seq_first_ctx_word_embeds = ctx_word_embeds.transpose(0, 1)         # seq_len x batch x embed_dim
        sequence_output, _ = rnn(seq_first_ctx_word_embeds)
        batch_first_output = sequence_output.transpose(0, 1)                # batch x seq_len x space_dim
        return batch_first_output


class DistanceAttention(nn.Module):
    def __init__(self, args, pos_embeds_rows, input_dims):
        super(DistanceAttention, self).__init__()

        # pos embeds
        self.manifold = gt.PoincareBall() if args.attn_metric == cs.HY else gt.Euclidean()
        with torch.no_grad():
            pos_embeds = init_embeddings(pos_embeds_rows, input_dims)
            if args.attn_metric == cs.HY:
                pos_embeds = pmath.expmap0(pos_embeds, k=self.manifold.k)
            beta = torch.Tensor(1).uniform_(-0.01, 0.01)
        self.position_embeds = gt.ManifoldParameter(pos_embeds, manifold=self.manifold)

        if args.attn_metric == cs.HY:
            self.key_dense = hnn.MobiusLinear(input_dims, input_dims, hyperbolic_input=True, hyperbolic_bias=True)
            self.query_dense = hnn.MobiusLinear(input_dims, input_dims, hyperbolic_input=True, hyperbolic_bias=True)
            self.addition = lambda x, y: pmath.mobius_add(x, y, k=self.manifold.k)
            self.distance_function = lambda x, y: pmath.dist(x, y, k=self.manifold.k)
            self.midpoint = hnn.weighted_mobius_midpoint
        else:
            self.key_dense = nn.Linear(input_dims, input_dims)
            self.query_dense = nn.Linear(input_dims, input_dims)
            self.addition = torch.add
            self.distance_function = utils.euclidean_distance
            self.midpoint = hnn.weighted_euclidean_midpoint

        self.encoder_to_attn_map = define_mapping(args.encoder_metric, args.attn_metric, args.c)
        self.attention_function = nn.Softmax(dim=1) if args.attn == "softmax" else nn.Sigmoid()
        self.beta = torch.nn.Parameter(beta, requires_grad=True)

    def forward(self, values, position_indexes):
        """
        :param values: batch x seq_len x input_dim
        :param position_indexes: batch x seq_len
        :return: batch x input_dim
        """
        values = self.encoder_to_attn_map(values)
        pos_embeds = self.position_embeds[position_indexes]             # b x seq_len x input_dim
        attn_embeds = self.addition(values, pos_embeds)                 # b x seq_len x input_dim

        queries = self.query_dense(attn_embeds)                         # b x seq_len x input_dim
        keys = self.key_dense(attn_embeds)                              # b x seq_len x input_dim

        attn_weights = self.get_attention_weights(queries, keys)        # b x seq_len
        return self.midpoint(values, attn_weights.unsqueeze(dim=2)), attn_weights

    def get_attention_weights(self, queries, keys):
        """
        :param queries: batch x seq_len x space_dim
        :param keys: batch x seq_len x space_dim
        """
        distances = self.distance_function(queries, keys)  # b x seq_len
        argument = -self.beta * distances
        return self.attention_function(argument)


class Model(nn.Module):

    def __init__(self, args, vocabs, word2vec=None):
        self.args = args

        super(Model, self).__init__()
        self.word_embed_manifold = gt.PoincareBall() if args.embedding_metric == cs.HY else gt.Euclidean()
        self.train_word_embeds = args.train_word_embeds == 1
        self.word_lut = self.init_lut(word2vec, len(vocabs[cs.TOKEN_VOCAB].label2wordvec_idx), args.word_emb_size)
        self.word_lut.requires_grad = self.train_word_embeds

        self.concat_dropout = nn.Dropout(p=args.concat_dropout)
        self.classif_dropout = nn.Dropout(p=args.classif_dropout)

        # encoders
        self.mention_encoder = MentionEncoder(vocabs[cs.CHAR_VOCAB], args)
        self.context_encoder = ContextEncoder(args)
        men_dim = self.mention_encoder.mention_output_dim
        char_dim = self.mention_encoder.char_output_dim
        ctx_dim = self.context_encoder.ctx_output_dim

        # ctx concat and attn
        ctx_concat_layer = hnn.MobiusConcat if args.encoder_metric == cs.HY else hnn.EuclConcat
        self.ctx_concat = ctx_concat_layer(ctx_dim * 2, ctx_dim)
        self.ctx_attn = DistanceAttention(args, args.context_len * 2 + 2, ctx_dim * 2)

        # full concat of mention and context
        input_classif_dim = men_dim + char_dim + ctx_dim * 2
        full_concat_layer = hnn.MobiusConcat if args.concat_metric == cs.HY else hnn.EuclConcat
        self.full_concat = full_concat_layer(input_classif_dim, men_dim, second_input_dim=ctx_dim * 2, third_input_dim=char_dim)

        # classifier
        classifier_layer = hnn.MobiusMLR if args.mlr_metric == cs.HY else hnn.EuclMLR
        self.classifier = classifier_layer(input_classif_dim, vocabs[cs.TYPE_VOCAB].size())

        self.attn_to_concat_map = define_mapping(args.attn_metric, args.concat_metric, args.c)
        self.concat_to_mlr_map = define_mapping(args.concat_metric, args.mlr_metric, args.c)

    def forward(self, input):
        context, ctx_position, ctx_len_tensor = input[0], input[1], input[2]
        mentions, mention_chars = input[3], input[4]

        # mention encoder
        mention_vectors, char_vectors = self.mention_encoder(mentions, mention_chars, self.word_lut)  # men: b x 2*space_dim

        # context encoder
        fwd_pass, bkwd_pass = self.context_encoder(context, self.word_lut)              # batch x ctx_len x space_dim
        ctx_concatenated = self.ctx_concat(fwd_pass, bkwd_pass)                         # b x ctx_len x 2*space_dim

        # context attention
        ctx_attn, attn_weights = self.ctx_attn(ctx_concatenated, ctx_position)          # b x 2*space_dim

        # concat all
        mention_vectors = self.concat_dropout(self.attn_to_concat_map(mention_vectors))
        char_vectors = self.concat_dropout(self.attn_to_concat_map(char_vectors))
        ctx_attn = self.concat_dropout(self.attn_to_concat_map(ctx_attn))
        text_vector = self.full_concat(mention_vectors, ctx_attn, third_input=char_vectors)     # b x output_dim

        # classifier
        text_vector = self.classif_dropout(self.concat_to_mlr_map(text_vector))
        logits = self.classifier(text_vector)                                           # batch x type_quantity

        return logits, attn_weights, text_vector.detach()

    def init_lut(self, weights, dim_0, dim_1):
        if weights is None:
            with torch.no_grad():
                weights = init_embeddings(dim_0, dim_1)
                if self.args.embedding_metric == cs.HY:
                    weights = pmath.expmap0(weights, k=self.word_embed_manifold.k)
        return gt.ManifoldParameter(weights, manifold=self.word_embed_manifold)

    def project_embeds(self):
        """Projects embeddings back into the hyperbolic ball, for numerical stability"""
        with torch.no_grad():
            if self.train_word_embeds and self.args.embedding_metric == cs.HY:
                self.word_lut.data = pmath.project(self.word_lut, k=self.word_embed_manifold.k)
            if self.args.attn_metric == cs.HY:
                k = self.ctx_attn.manifold.k
                self.ctx_attn.position_embeds.data = pmath.project(self.ctx_attn.position_embeds, k=k)
                self.mention_encoder.mention_attn.position_embeds.data = pmath.project(self.mention_encoder.mention_attn.position_embeds, k=k)

            self.mention_encoder.char_lut.data = pmath.project(self.mention_encoder.char_lut, k=self.mention_encoder.manifold.k)


def init_embeddings(dim_0, dim_1, k=0.0001):
    return torch.zeros((dim_0, dim_1), dtype=cs.DEFAULT_DTYPE, device=cs.DEVICE).uniform_(-k, k)

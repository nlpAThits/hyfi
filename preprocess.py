#!/usr/bin/env python
# encoding: utf-8

import argparse
from tqdm import tqdm
from os import path

import hyfi
from hyfi.constants import *
from hyfi.utils import process_line

DS = "distant_supervision"
CR = "crowd"
CR_TRAIN = f"{CR}/train_m.json"
CR_DEV = f"{CR}/dev.json"
CR_TEST = f"{CR}/test.json"
EL_TRAIN = f"{DS}/el_train.json"
EL_DEV = f"{DS}/el_dev.json"
HW_TRAIN = f"{DS}/headword_train.json"
HW_DEV = f"{DS}/headword_dev.json"

default_dtype = torch.float64
torch.set_default_dtype(default_dtype)
log = hyfi.utils.get_logging()


class Word2Vec:

    def __init__(self):
        self.token2vec = {}

    def add(self, token, vector):
        self.token2vec[token] = vector

    def get_vec(self, word):
        if word in self.token2vec:
            return self.token2vec[word]
        if word.lower() in self.token2vec:
            return self.token2vec[word.lower()]

        return self.get_unk_vector()

    def __contains__(self, word):
        return word in self.token2vec or word.lower() in self.token2vec

    def get_unk_vector(self):
        return self.token2vec[UNK_WORD]


def make_vocabs(args):
    """It creates a Dict for the words on the whole dataset, and the types"""
    token_vocab = hyfi.TokenDict(lower=True)
    type_vocab = hyfi.TypeDict()

    all_files = [path.join(args.dataset, fpath) for fpath in [EL_TRAIN, HW_TRAIN, CR_TRAIN, CR_DEV, CR_TEST]]
    bar = tqdm(desc="make_vocabs", total=hyfi.utils.wc(all_files))
    for data_file in all_files:
        for line in open(data_file, buffering=BUFFER_SIZE):
            bar.update()

            fields, tokens = process_line(line)

            for token in tokens:
                token_vocab.add(token)

            for mention_type in fields[TYPE]:
                type_vocab.add(mention_type)

    bar.close()

    char_vocab = hyfi.Dict()
    char_vocab.add(UNK_WORD)
    for char in CHARS:
        char_vocab.add(char)

    log.info("Created vocabs:\n\t#token: {}\n\t#type: {}\n\t#chars: {}".format(token_vocab.size(), type_vocab.size(),
                                                                               char_vocab.size()))

    return {TOKEN_VOCAB: token_vocab, TYPE_VOCAB: type_vocab, CHAR_VOCAB: char_vocab}


def make_word2vec(filepath, token_vocab):
    word2vec = Word2Vec()
    log.info(f"Start loading pretrained word vecs from {filepath}")
    for line in tqdm(open(filepath), total=hyfi.utils.wc(filepath)):
        fields = line.strip().split()
        embed_dim = 100 if len(fields) < 200 else 300
        token = fields[0]
        try:
            vec = list(map(float, fields[1:]))
            if len(vec) != embed_dim:
                raise ValueError
        except ValueError:
            log.info(f"Wrong parse: {token}")
            continue

        embedding = torch.Tensor(vec)

        if token == UNK_WORD:
            embedding /= 10

        word2vec.add(token, embedding)

    ret = []
    oov = 0

    # PAD word (index 0) is a vector full of zeros
    ret.append(torch.zeros(word2vec.get_unk_vector().size()))
    token_vocab.label2wordvec_idx[hyfi.constants.PAD_WORD] = 0

    for idx in range(1, token_vocab.size()):
        token = token_vocab.idx2label[idx]

        if token in word2vec:
            vec = word2vec.get_vec(token)
            token_vocab.label2wordvec_idx[token] = len(ret)
            ret.append(vec)
        else:
            oov += 1

    ret = torch.stack(ret)          # creates a "matrix" of token.size() x embed_dim

    norm_bigger_than_one = (ret.norm(dim=1) > 1).sum()
    log.info(f"Amount of word embeddings with norm > 1: {norm_bigger_than_one}")
    log.info("* OOV count: %d" %oov)
    log.info("* Embedding size (%s)" % (", ".join(map(str, list(ret.size())))))
    return ret


def make_data(data_files, vocabs, type_quantity, args):
    data = []
    for fname in data_files:
        file_path = path.join(args.dataset, fname)
        for line in tqdm(open(file_path, buffering=BUFFER_SIZE), total=hyfi.utils.wc(file_path)):
            fields, _ = process_line(line)

            mention = hyfi.Mention(fields)
            data.append(mention)

    log.info(f"Prepared {len(data)} mentions.")
    dataset = hyfi.Dataset(data, args, type_quantity)

    log.info(f"Transforming to matrix {len(data)} mentions from {data_files}")
    dataset.to_matrix(vocabs, args)

    return dataset


def main(args):
    log.info("Preparing vocabulary...")
    vocabs = make_vocabs(args)
    type_quantity = len(vocabs[TYPE_VOCAB].label2idx)

    log.info("Preparing word vectors...")
    word2vec = make_word2vec(args.word2vec, vocabs[TOKEN_VOCAB])

    log.info("Preparing training...")
    train = make_data([CR_TRAIN, EL_TRAIN, HW_TRAIN], vocabs, type_quantity, args)
    log.info("Preparing crowd training...")
    crowd_train = make_data([CR_TRAIN], vocabs, type_quantity, args)
    log.info("Preparing dev...")
    dev = make_data([CR_DEV], vocabs, type_quantity, args)
    log.info("Preparing test...")
    test = make_data([CR_TEST], vocabs, type_quantity, args)

    log.info("Saving pretrained word vectors to '%s'..." % (args.save_data + "/word2vec.pt"))
    torch.save(word2vec, args.save_data + "/word2vec.pt")

    log.info("Saving data to '%s'..." % (args.save_data + "/data.pt"))
    save_data = {"vocabs": vocabs, "train": train, "crowd_train": crowd_train, "dev": dev, "test": test}
    torch.save(save_data, args.save_data + "/data.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess.py")

    # Input data
    parser.add_argument("--dataset", required=True, help="Path to the dataset")
    parser.add_argument("--word2vec", required=True, type=str, help="Path to pretrained word vectors.")
    # Mention
    parser.add_argument("--mention_length", default=5, type=int,
                        help="Max amount of words taken for mention representation")
    parser.add_argument("--mention_char_length", default=10, type=int,
                        help="Max amount of chars taken for mention representation")
    # Context
    parser.add_argument("--context_length", default=25, type=int,
                        help="Max length of the context on each side (left or right)")
    # Output data
    parser.add_argument("--save_data", required=True, help="Path to the output data.")

    args = parser.parse_args()
    hyfi.utils.set_seed(42)
    main(args)

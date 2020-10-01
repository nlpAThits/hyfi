#!/usr/bin/env python
# encoding: utf-8

import torch
import hyfi.constants as c


class Mention(object):

    def __init__(self, fields):
        self.fields = fields

    def preprocess(self, vocabs, args):
        self.vocabs = vocabs
        self.max_context_len = args.context_length
        self.max_mention_len = args.mention_length
        self.max_mention_char_len = args.mention_char_length

        self.types = self.type_idx()        # type index in vocab
        self.mention = self.get_mention_idx()
        self.mention_chars = self.get_mention_chars()
        self.context = self.get_context_idx()
        self.context_positions = self.get_context_positions_idx()

    def get_mention_idx(self):
        head = self.fields[c.MENTION].split()[:self.max_mention_len]
        if not head:
            return torch.LongTensor([c.PAD])
        return self.vocabs[c.TOKEN_VOCAB].convert_to_idx(head, c.UNK_WORD)

    def get_context_idx(self):
        """Simple heuristic which will truncate one token at a time from each side of the context sentence, trying to
        keep the mention in the center and the longest possible context information, within the 'max_content_len'."""
        left_len = len(self.fields[c.LEFT_CTX])
        mention_words = self.fields[c.MENTION].split()
        if len(mention_words) > self.max_context_len:
            self.mention_ini = 0
            self.mention_end = self.max_context_len - 1
            return self.vocabs[c.TOKEN_VOCAB].convert_to_idx(mention_words[:self.max_context_len], c.UNK_WORD)

        ctx_words = self.fields[c.LEFT_CTX] + mention_words + self.fields[c.RIGHT_CTX]
        mention_ini, mention_end = left_len, left_len + len(mention_words) - 1

        if not ctx_words:
            return torch.LongTensor([c.PAD])

        while len(ctx_words) > self.max_context_len:
            if mention_ini > len(ctx_words) - (mention_end + 1):
                del ctx_words[0]
                mention_ini -= 1
                mention_end -= 1
            else:
                del ctx_words[-1]
        self.mention_ini = mention_ini
        self.mention_end = mention_end
        return self.vocabs[c.TOKEN_VOCAB].convert_to_idx(ctx_words, c.UNK_WORD)

    def get_context_positions_idx(self):
        """
        We use different ids for left and right context positions.
        Left is in the range [2:max_left_len]
        Mention is 1's
        Right is in the range: [max_left_len + 2, max_right_len]

        Example: if max_left_len = max_right_len = 15,
        Sentence: "During the year of 1921 _Pablo Picasso_ painted large cubist compositions", mention: "Pablo Picasso"
        Pos_idx:     6     5   4   3   2     1       1        17     18    19        20
        This ids are just the entry id in a lookup table of position embeddings.
        """
        mention_len = self.mention_end - self.mention_ini + 1
        left_len = self.mention_ini
        right_len = len(self.context) - (self.mention_end + 1)

        left_pos = [i for i in range(2, left_len + 2)][::-1]
        mention_pos = [1] * mention_len
        right_pos = [i for i in range(self.max_context_len + 2, self.max_context_len + 2 + right_len)]

        return torch.LongTensor(left_pos + mention_pos + right_pos)

    def type_idx(self):
        types = []
        for mention_type in self.fields[c.TYPE]:
            types.append(self.vocabs[c.TYPE_VOCAB].lookup(mention_type))
        return torch.LongTensor(types)

    def get_mention_chars(self):
        chars = self.fields[c.MENTION][:self.max_mention_char_len]
        if not chars:
            return torch.LongTensor([c.PAD])
        return self.vocabs[c.CHAR_VOCAB].convert_to_idx(chars, c.UNK_WORD)

    def type_len(self):
        return len(self.fields[c.TYPE])

    def clear(self):
        del self.fields
        del self.mention
        del self.mention_chars
        del self.left_context
        del self.right_context
        del self.types





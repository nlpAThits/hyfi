#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

import math
from random import shuffle
import torch
from torch.autograd import Variable

import hyfi
from hyfi.constants import PAD

from tqdm import tqdm

log = hyfi.utils.get_logging()


class Dataset(object):

    def __init__(self, data, args, type_quantity):
        self.args = args
        self.type_quantity = type_quantity
        self.device = None

        self.buckets = {}
        for mention in data:
            type_amount = 1     # mention.type_len()
            if type_amount in self.buckets:
                self.buckets[type_amount].append(mention)
            else:
                self.buckets[type_amount] = [mention]

    def to_matrix(self, vocabs, args):
        self.matrixes = {}
        for type_len, mentions in self.buckets.items():
            self.matrixes[type_len] = self._bucket_to_matrix(mentions, vocabs, args)
        del self.buckets

    def _bucket_to_matrix(self, mentions, vocabs, args):
        """
        Creates tensors: ctx, ctx_pos, ctx_len, mentions ids, mention chars and types

        Used only on PREPROCESSING time
        """
        bucket_size = len(mentions)

        ctx_tensor = torch.LongTensor(bucket_size, args.context_length).fill_(PAD)
        ctx_position = torch.LongTensor(bucket_size, args.context_length).fill_(PAD)
        ctx_len_tensor = torch.LongTensor(bucket_size)
        mention_tensor = torch.LongTensor(bucket_size, args.mention_length).fill_(PAD)
        mention_char_tensor = torch.LongTensor(bucket_size, args.mention_char_length).fill_(PAD)
        type_list = []

        bar = tqdm(desc="to_matrix", total=bucket_size)

        for i in range(bucket_size):
            bar.update()
            item = mentions[i]
            item.preprocess(vocabs, args)

            ctx_tensor[i].narrow(0, 0, item.context.size(0)).copy_(item.context)

            ctx_position[i].narrow(0, 0, item.context_positions.size(0)).copy_(item.context_positions)
            ctx_len_tensor[i] = item.context.size(0)

            mention_tensor[i].narrow(0, 0, item.mention.size(0)).copy_(item.mention)
            mention_char_tensor[i].narrow(0, 0, item.mention_chars.size(0)).copy_(item.mention_chars)
            type_list.append(item.types)

        bar.close()

        return [t.contiguous() for t in [ctx_tensor, ctx_position, ctx_len_tensor, mention_tensor,
                                         mention_char_tensor]], type_list

    def get_mention_len(self):
        return self.matrixes[1][0][3].size(1)

    def get_context_len(self):
        return self.matrixes[1][0][0].size(1)

    def __len__(self):
        try:
            return self.num_batches
        except AttributeError as e:
            raise AttributeError("Dataset.set_batch_size must be invoked before to calculate the length") from e

    def shuffle(self):
        shuffle(self.iteration_order)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.num_batches = 0
        self.iteration_order = []
        for type_len, (tensors, type_ids) in self.matrixes.items():
            len_tensor = len(tensors[0])
            bucket_num_batches = math.ceil(len_tensor / batch_size)
            for i in range(bucket_num_batches):
                start_index = batch_size * i
                end_index = batch_size * (i + 1) if batch_size * (i + 1) < len_tensor else len_tensor
                self.iteration_order.append((type_len, start_index, end_index))

            self.num_batches += bucket_num_batches

    def create_one_hot_types(self, batch_type_ids):
        one_hot_vectors = torch.zeros((len(batch_type_ids), self.type_quantity), device=self.device, dtype=torch.float32)
        for i in range(len(batch_type_ids)):
            one_hot_vectors[i][batch_type_ids[i]] = 1.0

        return one_hot_vectors.contiguous()

    def __getitem__(self, index):
        """
        :param index:
        :return: Matrices of different parts (head string, contexts, types) of every instance
        """
        bucket, start_index, end_index = self.iteration_order[index]

        tensors, type_ids = self.matrixes[bucket]

        batch_tensors = [self.process_batch(tensor, start_index, end_index) for tensor in tensors]
        batch_type_ids = [t.to(self.device) for t in type_ids[start_index:end_index]]

        # one-hot matrix takes a lot of spaces so it creates it on the fly for each batch
        one_hot_types = self.create_one_hot_types(batch_type_ids)
        return batch_tensors + [batch_type_ids, Variable(one_hot_types)]

    def process_batch(self, data_tensor, start_index, end_index):
        batch_data = data_tensor[start_index: end_index]
        return self.to_cuda(batch_data).contiguous()

    def to_cuda(self, batch_data):
        batch_data = batch_data.to(self.device)
        return Variable(batch_data)

    def subsample(self, length=None):
        """
        :param length: of the subset. If None, then length is one batch size, at most.
        :return: shuffled subset of self.
        """
        if not length:
            length = self.batch_size

        other = Dataset([], self.args, self.type_quantity)

        other.matrixes = {}
        for type_len, tensors in self.matrixes.items():
            other.matrixes[type_len] = [tensor[:length] for tensor in tensors]

        other.set_batch_size(self.batch_size)

        return other

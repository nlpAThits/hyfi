#!/usr/bin/env python
# encoding: utf-8


import sys
import logging
import random
import json
import numpy as np
import torch
from . import constants


def set_seed(seed):
    """Sets random seed everywhere."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_logging(level=logging.DEBUG):
    log = logging.getLogger(__name__)
    if log.handlers:
        return log
    log.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log


def wc(files):
    if not isinstance(files, list) and not isinstance(files, tuple):
        files = [files]
    return sum([sum([1 for _ in open(fp, buffering=constants.BUFFER_SIZE)]) for fp in files])


def process_line(line):
    fields = json.loads(line)
    tokens = build_full_sentence(fields)
    return fields, tokens


def build_full_sentence(fields):
    return fields[constants.LEFT_CTX] + fields[constants.MENTION].split() + fields[constants.RIGHT_CTX]


def to_sparse(tensor):
    """Given a one-hot encoding vector returns a list of the indexes with nonzero values"""
    return torch.nonzero(tensor)


def expand_tensor(tensor, length):
    """
    :param tensor: dim: N x M
    :param length: l
    :return: tensor of (N * l) x M with every row intercalated and extended l times
    """
    rows, cols = tensor.size()
    repeated = tensor.repeat(1, length)
    return repeated.view(rows * length, cols)


def euclidean_dot_product(u, v):
    return (u * v).sum(dim=1)


# def plot_k(name, full_type_positions, full_closest_true_neighbor):
#     import matplotlib.pyplot as plt
#     plt.switch_backend('agg')
#     save_plot(plt, full_type_positions, f"img/{name}_full_type.png")
#     save_plot(plt, full_type_positions, f"img/{name}_full_type_cumulative.png", cumulative=True, density=True)
#     save_plot(plt, full_closest_true_neighbor, f"img/{name}_full_closest.png")
#     save_plot(plt, full_closest_true_neighbor, f"img/{name}_full_closest_cumulative.png", cumulative=True, density=True)


def save_plot(plt, data, filename, cumulative=False, density=False):
    plt.hist(data, cumulative=cumulative, density=density, bins=50)
    plt.savefig(filename)
    plt.clf()

def euclidean_distance(u, v):
    return torch.norm(u - v, dim=-1, keepdim=False)


#!/usr/bin/env python
# encoding: utf-8

# script to export type embeddings to the Tensorboard projector format (tsv)
# usage: python export_embeds.py --data=data/prep/poin-ctx25 --export_path=models/exported_model.pt --metric=hyperbolic

import torch
from torch.nn import DataParallel
import argparse
from train import config_parser
import hyfi.constants as cs
from hyfi.utils import get_logging
from hyfi.models import Model


parser = argparse.ArgumentParser("export_embeds.py")
config_parser(parser)
parser.add_argument("--words", default=1000, type=int, help="Amount of words to export. It can be 0 for no export")
args = parser.parse_args()
args.hyperbolic = args.metric == "hyperbolic"

log = get_logging()


def main():
    log.debug("Loading data from '%s'." % args.data)
    data = torch.load(args.data + "/data.pt")
    vocabs = data["vocabs"]
    type_vocab = vocabs[cs.TYPE_VOCAB]
    token_vocab = vocabs[cs.TOKEN_VOCAB]

    state_dict = torch.load(args.export_path, map_location="cpu")

    # log.debug("Building model...")
    # model = Model(args, vocabs)
    # model = DataParallel(model)
    # model.load_state_dict(state_dict)
    # model.to(cs.DEVICE)

    types = state_dict["module.classifier.p_k"]
    export_types(types, type_vocab)

    pos_embeds = state_dict["module.hyper_attn.position_embeds"]
    export_position_embeddings(pos_embeds)

    # word_lut = state_dict["module.word_lut"]
    # export_words(word_lut, token_vocab)
    log.info("Done!!")


def export_types(types, type_vocab):
    vecs, meta = [], ["label\tgran"]
    for i in range(len(types)):
        type_vec = "\t".join(map(str, types[i].tolist()))
        type_label = type_vocab.get_label(i)
        if type_label in cs.COARSE:
            gran = "coarse"
        elif type_label in cs.FINE:
            gran = "fine"
        else:
            gran = "ultra"

        vecs.append(type_vec)
        meta.append(f"{type_label}\t{gran}")
    export_name = args.export_path.split("/")[-1]
    export(f"export/{export_name}-pk-vecs.tsv", vecs)
    export(f"export/{export_name}-pk-meta.tsv", meta)


def export_position_embeddings(pos_embeds):
    vecs, meta = [], []
    for i in range(len(pos_embeds)):
        embed = "\t".join(map(str, pos_embeds[i].tolist()))

        vecs.append(embed)
        meta.append(str(i))

    export_name = args.export_path.split("/")[-1]
    export(f"export/{export_name}-pos_embeds.tsv", vecs)
    export(f"export/{export_name}-pos_embeds-meta.tsv", meta)


def export_words(word_lut, token_vocab):
    if args.words <= 0:
        return

    word_limit = min(args.words, token_vocab.size_of_word2vecs())

    vecs, meta = [], []
    for i in range(word_limit):
        word_vec = "\t".join(map(str, word_lut[i].tolist()))
        word_label = token_vocab.get_label(i)

        vecs.append(word_vec)
        meta.append(word_label)
    export_name = args.export_path.split("/")[-1]
    export(f"export/{export_name}-wordsvecs.tsv", vecs)
    export(f"export/{export_name}-wordsmeta.tsv", meta)


def export(path, data):
    with open(path, "w") as fp:
        for res in data:
            fp.write(res + "\n")


if __name__ == "__main__":
    main()

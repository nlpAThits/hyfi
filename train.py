#!/usr/bin/env python
# encoding: utf-8
from __future__ import division

import argparse
import random
import torch
import geoopt as gt
from torch.nn import DataParallel
import hyfi.constants as cs
from hyfi.models import Model, define_mapping
from hyfi.runner import Runner
from hyfi import utils


def config_parser(parser):
    # Data options
    parser.add_argument("--data", required=True, type=str, help="Data path.")

    # Sentence-level context parameters
    parser.add_argument("--men_nonlin", default="tanh", type=str, help="Non-linearity in mention encoder")
    parser.add_argument("--ctx_nonlin", default="tanh", type=str, help="Non-linearity in context encoder")
    parser.add_argument("--num_layers", default=1, type=int, help="Number of layers in MobiusGRU")
    parser.add_argument("--space_dims", default=20, type=int, help="Space dims.")

    # Component metrics
    parser.add_argument("--embedding_metric", default=cs.HY, type=str, help="hyperbolic | euclidean")
    parser.add_argument("--encoder_metric", default=cs.HY, type=str, help="hyperbolic | euclidean")
    parser.add_argument("--attn_metric", default=cs.HY, type=str, help="hyperbolic | euclidean")
    parser.add_argument("--concat_metric", default=cs.HY, type=str, help="hyperbolic | euclidean")
    parser.add_argument("--mlr_metric", default=cs.HY, type=str, help="hyperbolic | euclidean")

    # Other parameters
    parser.add_argument("--input_dropout", default=0.3, type=float, help="Dropout over input.")
    parser.add_argument("--concat_dropout", default=0.2, type=float, help="Dropout in concat.")
    parser.add_argument("--classif_dropout", default=0.0, type=float, help="Dropout in classifier.")
    parser.add_argument("--crowd_cycles", default=5, type=int, help="Number of crowd re-train.")
    parser.add_argument("--learning_rate", default=0.0005, type=float, help="Starting learning rate.")
    parser.add_argument("--weight_decay", default=0.00, type=float, help="L2 Regularization.")
    parser.add_argument("--batch_size", default=900, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=30, type=int, help="Number of training epochs.")
    parser.add_argument("--max_grad_norm", default=5, type=float,
                        help="If the norm of the gradient vector exceeds this, renormalize it to max_grad_norm")
    parser.add_argument("--patience", default=50, type=int, help="Patience for lr scheduler")
    parser.add_argument("--export_path", default="", type=str, help="Name of model to export")
    parser.add_argument("--export_epochs", default=20, type=int, help="Export every n epochs")
    parser.add_argument("--log_epochs", default=4, type=int, help="Log examples every n epochs")
    parser.add_argument("--load_model", default="", type=str, help="Path of model to load")
    parser.add_argument("--train_word_embeds", default=0, type=int, help="Wether to train word embeds or not")
    parser.add_argument("--seed", default=-1, type=int, help="Seed")
    parser.add_argument("--c", default=1.0, type=float, help="c param to project embeddings")
    parser.add_argument("--attn", default="softmax", type=str, help="Options: sigmoid | softmax")


parser = argparse.ArgumentParser("train.py")
config_parser(parser)
args = parser.parse_args()

seed = args.seed if args.seed > 0 else random.randint(1, 1000000)
utils.set_seed(seed)

log = utils.get_logging()
log.debug(args)


def get_dataset(data, args, key):
    dataset = data[key]
    dataset.set_batch_size(args.batch_size)
    dataset.shuffle()
    dataset.device = cs.DEVICE
    return dataset


def main():
    # Load data.
    log.debug(f"Loading data from {args.data }/data.pt")
    data = torch.load(args.data + "/data.pt")
    vocabs = data["vocabs"]

    # dataset splits
    train_data = get_dataset(data, args, "train")
    crowd_train_data = get_dataset(data, args, "crowd_train")
    dev_data = get_dataset(data, args, "dev")
    test_data = get_dataset(data, args, "test")

    args.mention_len = train_data.get_mention_len()
    args.context_len = train_data.get_context_len()

    if not args.load_model:
        log.debug(f"Loading word2vec from {args.data}/word2vec.pt")
        word2vec = torch.load(args.data + "/word2vec.pt")
        args.word_emb_size = word2vec.size(1)

        embed_mapping = define_mapping(args.embedding_metric, args.encoder_metric, args.c)
        log.debug(f"Embed mapping: Applying {embed_mapping} with c={args.c}")

        log.debug("Building model...")
        model = Model(args, vocabs, embed_mapping(word2vec))
        model = DataParallel(model)

    else:
        log.debug(f"Loading model from {args.load_model}")
        state_dict = torch.load(args.load_model)
        args.word_emb_size = state_dict["module.word_lut"].size(1)

        model = Model(args, vocabs)
        model = DataParallel(model)
        model.load_state_dict(state_dict)

    log.info(f"GPU's available: {torch.cuda.device_count()}")
    model.to(cs.DEVICE)

    optim = gt.optim.RiemannianAdam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                                    stabilize=5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', factor=0.5, patience=args.patience,
                                                           verbose=True)

    n_params = sum([p.nelement() for p in model.parameters() if p.requires_grad])
    log.debug(f"Number of parameters: {n_params}")

    runner = Runner(model, optim, scheduler, vocabs, train_data, crowd_train_data, dev_data, test_data, args)

    # Train.
    log.info("Start training...")
    runner.train()
    log.info("Done!")


if __name__ == "__main__":
    main()

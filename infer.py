#!/usr/bin/env python
# encoding: utf-8

import torch
from torch.nn import DataParallel
import argparse
from hyfi.models import Model
from hyfi.runner import Runner
from hyfi.constants import DEVICE
from hyfi.utils import get_logging
from train import config_parser, get_dataset


parser = argparse.ArgumentParser("infer.py")
config_parser(parser)
args = parser.parse_args()

log = get_logging()


def main():
    log.debug("Loading data from '%s'." % args.data)
    data = torch.load(args.data + "/data.pt")
    vocabs = data["vocabs"]

    dev_data = get_dataset(data, args, "dev")
    test_data = get_dataset(data, args, "test")

    state_dict = torch.load(args.export_path)
    args.word_emb_size = state_dict["module.word_lut"].size(1)
    args.mention_len = dev_data.get_mention_len()
    args.context_len = dev_data.get_context_len()

    log.debug("Building model...")
    model = Model(args, vocabs)
    model = DataParallel(model)
    model.load_state_dict(state_dict)
    model.to(DEVICE)

    runner = Runner(model, None, None, vocabs, None, None, dev_data, test_data, args)

    log.info("INFERENCE ON DEV DATA:")
    runner.instance_printer.show(dev_data)
    runner.print_full_validation(dev_data, "DEV")

    log.info("\n\nINFERENCE ON TEST DATA:")
    runner.instance_printer.show(test_data)
    runner.print_full_validation(test_data, "TEST")


if __name__ == "__main__":
    main()

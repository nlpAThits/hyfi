#!/usr/bin/env python
# encoding: utf-8

import copy
import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from geoopt.manifolds.stereographic.math import dist0 as hyperbolic_norm

from hyfi.utils import get_logging
from hyfi.predictor import assign_types, assign_exactly_k_types
from hyfi.evaluate import evaluate, raw_evaluate, stratified_evaluate
from hyfi.constants import TYPE_VOCAB, DEVICE
from hyfi.instance_printer import InstancePrinter
from hyfi.loss import MultiTaskBCELoss

log = get_logging()


class Runner:

    def __init__(self, model, optim, scheduler, vocabs, train_data, crowd_train_data, dev_data, test_data, args):
        self.model = model
        self.optim = optim
        self.scheduler = scheduler
        self.vocabs = vocabs
        self.train_data = train_data
        self.crowd_train_data = crowd_train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.loss = MultiTaskBCELoss(vocabs)
        self.args = args
        self.instance_printer = InstancePrinter(vocabs, self.model)
        self.writer = SummaryWriter(f"tensorboard/{args.export_path}")

    def train(self):
        log.debug(self.model)

        max_macro_f1, best_model_state, best_epoch = -1, None, 0

        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_epoch(epoch)

            # extra iterations to fine-tune on crowdsourced data
            for i in range(self.args.crowd_cycles):
                self.train_crowd_data()

            dev_results, dev_loss = self.validate_typing(self.dev_data, "dev", epoch)
            dev_macro_f1 = dev_results[1][2]
            lr = self.optim.param_groups[0]["lr"]

            log.info(f"\n\nResults ep {epoch}: tr loss: {train_loss * 100:.2f}; "
                     f"Dev loss: {dev_loss * 100:.2f}, MaF1 {dev_macro_f1:.2f}, lr: {lr}")

            self.scheduler.step(dev_macro_f1)

            self.writer.add_scalar("lr", float(lr), epoch)
            self.writer.add_scalar("train/loss", train_loss, epoch)
            self.writer.add_scalar("dev/loss", dev_loss, epoch)
            self.writer.add_scalar("dev/macro_f1", dev_macro_f1, epoch)

            if dev_macro_f1 > max_macro_f1:
                max_macro_f1 = dev_macro_f1
                best_model_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                log.info(f"* Best coarse macro F1 {dev_macro_f1:0.2f} at epoch {epoch} *")

            self.print_full_validation(self.dev_data, "dev")

            if self.args.export_path and epoch % self.args.export_epochs == 0:
                model_path = f"models/{self.args.export_path}-{epoch}ep.pt"
                log.info(f"-- Exporting model to {model_path}")
                torch.save(self.model.state_dict(), model_path)

            if epoch % self.args.log_epochs == 0:
                self.instance_printer.show(self.dev_data, n=1)

            for name, param in self.model.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

        best_model_path = f"models/{self.args.export_path}-{best_epoch}bstep.pt"
        log.info(f"-- Exporting b3st model to {best_model_path}")
        self.model.load_state_dict(best_model_state)
        torch.save(self.model.state_dict(), best_model_path)

        log.info(f"Final evaluation on best Macro F1 ({max_macro_f1:0.3f}) from epoch {best_epoch}")
        self.instance_printer.show(self.dev_data, n=10)
        self.print_full_validation(self.dev_data, "dev")
        self.print_full_validation(self.test_data, "test")

        self.writer.close()

    def train_epoch(self, epoch):
        """:param epoch: int >= 1"""
        self.train_data.shuffle()
        total_loss, avg_grad_norm, avg_text_euclid_norm, avg_text_hyperbolic_norm = [], [], [], []
        self.model.train()
        for i in tqdm(range(len(self.train_data)), desc="train_epoch_{}".format(epoch)):
            batch = self.train_data[i]
            one_hot_true_types = batch[6]

            self.model.module.project_embeds()

            self.optim.zero_grad()
            logits, _, text_vectors = self.model(batch)
            loss = self.loss.calculate_loss(logits, one_hot_true_types)
            loss = loss.mean()                  # because of DataParallel
            loss.backward()

            self.write_total_grad_norm(i, epoch)

            if self.args.max_grad_norm >= 0:
                clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optim.step()

            total_loss.append(loss.item())

            # write norm of last layer
            gradient = self.model.module.classifier.a_k.grad
            if gradient is not None:
                grad_norm = gradient.data.norm(2).item()
                avg_grad_norm.append(grad_norm)
                self.writer.add_scalar(f"classif_grad_norm/epoch_{epoch}", grad_norm, i)

            # write text vector norms
            avg_text_euclid_norm += text_vectors.norm(2, dim=1).tolist()
            avg_text_hyperbolic_norm += hyperbolic_norm(text_vectors, k=torch.Tensor([-1.0]).to(DEVICE)).tolist()

        self.writer.add_scalar("classif_grad_norm/avg_norm", np.mean(avg_grad_norm), epoch)
        self.writer.add_scalar("text_vector/euclid_norm", np.mean(avg_text_euclid_norm), epoch)
        self.writer.add_scalar("text_vector/hyperb_norm", np.mean(avg_text_hyperbolic_norm), epoch)

        return np.mean(total_loss)

    def train_crowd_data(self):
        data = self.crowd_train_data
        data.shuffle()
        self.model.train()
        for i in tqdm(range(len(data)), desc="train_crowd_data"):
            batch = data[i]
            one_hot_true_types = batch[6]

            self.model.module.project_embeds()

            self.optim.zero_grad()
            logits, _, _ = self.model(batch)
            loss = self.loss.calculate_loss(logits, one_hot_true_types)
            loss = loss.mean()                 # because of DataParallel
            loss.backward()

            if self.args.max_grad_norm >= 0:
                clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optim.step()

    def write_total_grad_norm(self, batch_number, epoch):
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is None or p.grad.data is None: continue
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.writer.add_scalar(f"grad_total_norm/epoch_{epoch}", total_norm, batch_number)

    def validate_typing(self, data, name, epoch):
        true_and_preds, loss = self.infer_results(data, name, epoch)
        result = raw_evaluate(true_and_preds)
        return result, loss

    def infer_results(self, data, name, epoch, precision_at=0):
        total_loss, results = [], []
        self.model.eval()
        sigmoid = torch.nn.Sigmoid()
        with torch.no_grad():
            for i in tqdm(range(len(data)), desc=f"infer_{name}_ep{epoch}_P@{precision_at}"):
                batch = data[i]
                type_indexes, one_hot_true_types = batch[-2], batch[6]

                logits, _, _ = self.model(batch)
                probability_predictions = sigmoid(logits)
                loss = self.loss.calculate_loss(logits, one_hot_true_types)
                loss = loss.mean()

                total_loss.append(loss.item())
                if precision_at > 0:
                    results += assign_exactly_k_types(probability_predictions, type_indexes,
                                                      self.vocabs[TYPE_VOCAB], precision_at=precision_at)
                else:
                    results += assign_types(probability_predictions, type_indexes)

            return results, np.mean(total_loss)

    def print_full_validation(self, dataset, name):
        log.info(f"FULL VALIDATION ON {name.upper()}")
        true_and_preds, _ = self.infer_results(dataset, name, -1)

        full_eval = evaluate(true_and_preds)
        stratified_eval = stratified_evaluate(true_and_preds, self.vocabs[TYPE_VOCAB])

        log.info("Strict (p,r,f1), Macro (p,r,f1), Micro (p,r,f1)")
        strat_string = "\n".join([item for pair in zip(["COARSE", "FINE", "ULTRAFINE"], stratified_eval) for item in pair])
        log.info(f"Total:\n{full_eval}")
        log.info(strat_string)
        # self.print_precision_at(dataset, name)
        for_export = "".join(stratified_eval)
        log.info(f"\n{for_export}")

    def print_precision_at(self, dataset, name):
        true_and_pred_at_1, _ = self.infer_results(dataset, name, -1, precision_at=1)
        res_co_at_1, res_fi_at_1, res_uf_at_1 = stratified_evaluate(true_and_pred_at_1, self.vocabs[TYPE_VOCAB])

        true_and_pred_at_3, _ = self.infer_results(dataset, name, -1, precision_at=3)
        _, _, res_uf_at_3 = stratified_evaluate(true_and_pred_at_3, self.vocabs[TYPE_VOCAB])

        true_and_pred_at_5, _ = self.infer_results(dataset, name, -1, precision_at=5)
        _, _, res_uf_at_5 = stratified_evaluate(true_and_pred_at_5, self.vocabs[TYPE_VOCAB])

        out = f"PRECISION AT N\nCoarse@1\n{res_co_at_1}\nFine@1\n{res_fi_at_1}\n" \
              f"UltraFine@1\n{res_uf_at_1}\nUltraFine@3\n{res_uf_at_3}\nUltraFine@5\n{res_uf_at_5}\n"
        log.info(out)

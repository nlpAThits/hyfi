
import torch
from torch.nn import BCEWithLogitsLoss
from hyfi.constants import TYPE_VOCAB


class MultiTaskBCELoss:
    def __init__(self, vocabs):
        self.loss_function = BCEWithLogitsLoss()
        type_vocab = vocabs[TYPE_VOCAB]
        self.coarse_slice = len(type_vocab.get_coarse_ids())
        self.fine_slice = self.coarse_slice + len(type_vocab.get_fine_ids())

    def calculate_loss(self, logits, one_hot_true_labels):
        """Calculates multitask loss introduced in 'Ultra-Fine Entity Typing' by Choi et al."""
        loss = 0.0
        loss += self.calculate_loss_by_granularity(logits, one_hot_true_labels, end=self.coarse_slice)
        loss += self.calculate_loss_by_granularity(logits, one_hot_true_labels, ini=self.coarse_slice,
                                                   end=self.fine_slice)
        loss += self.calculate_loss_by_granularity(logits, one_hot_true_labels, ini=self.fine_slice)
        return loss

    def calculate_loss_by_granularity(self, logits, one_hot_true_types, ini=0, end=None):
        gran_targets = one_hot_true_types[:, ini:end]
        gran_target_sum = torch.sum(gran_targets, 1)

        if torch.sum(gran_target_sum.data) <= 0:  # if there is no target in this granularity in this batch
            return 0.

        comparison_tensor = torch.FloatTensor([1.0]).to(one_hot_true_types.device)
        gran_mask = torch.min(gran_target_sum.data, comparison_tensor).nonzero().squeeze(dim=1)
        gran_logit_masked = logits[:, ini:end][gran_mask, :].float()
        gran_target_masked = gran_targets[gran_mask]
        return self.loss_function(gran_logit_masked, gran_target_masked)

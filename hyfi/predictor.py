
from hyfi.utils import get_logging
import torch

log = get_logging()


def assign_types(probability_predictions, type_indexes, threshold=0.5):
    """
    :param probability_predictions: batch x total_type_len
    :param type_indexes: batch x type_len
    :return: list of pairs of true type indexes and predicted type indexes
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = []
    for i in range(len(probability_predictions)):
        predictions = probability_predictions[i]
        predicted_indexes = (predictions >= threshold).nonzero().flatten().tolist()
        if len(predicted_indexes) == 0:
            predicted_indexes = [predictions.max(0)[1].item()]

        results.append([type_indexes[i], torch.LongTensor(predicted_indexes).to(device)])

    return results


def assign_exactly_k_types(probability_predictions, type_indexes, type_dict, precision_at):
    """
    It assigns the top precision_at predictions per granularity
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    coarse_slice = len(type_dict.get_coarse_ids())
    fine_slice = coarse_slice + len(type_dict.get_fine_ids())

    coarse_predictions = probability_predictions[:, :coarse_slice]
    fine_predictions = probability_predictions[:, coarse_slice:fine_slice]
    ultrafine_predictions = probability_predictions[:, fine_slice:]

    results = []
    for i in range(len(probability_predictions)):
        co_pred, fi_pred, uf_pred = coarse_predictions[i], fine_predictions[i], ultrafine_predictions[i]

        predicted_indexes = co_pred.topk(precision_at)[1].tolist()
        predicted_indexes += (fi_pred.topk(precision_at)[1] + coarse_slice).tolist()
        predicted_indexes += (uf_pred.topk(precision_at)[1] + fine_slice).tolist()

        results.append([type_indexes[i], torch.LongTensor(predicted_indexes).to(device)])

    return results

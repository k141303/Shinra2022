import torch


from utils.scoring_utils import calc_f1


def classification_micro_f1(outputs, labels, prefix="train"):
    outputs = outputs["classification"] >= 0.5
    labels = labels["classification"]

    cnt = {}
    cnt["TP"] = torch.logical_and(outputs, labels).sum().item()
    cnt["TPFP"] = outputs.sum().item()
    cnt["TPFN"] = labels.sum().item()

    scores = calc_f1(cnt, prefix=prefix)
    return scores, scores[f"{prefix}_F1"]

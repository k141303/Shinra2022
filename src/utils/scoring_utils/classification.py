import torch


def calc_f1(cnt, prefix="train"):
    if not cnt["TP"]:
        return {
            f"{prefix}_Prec": 0.0,
            f"{prefix}_Rec": 0.0,
            f"{prefix}_F1": 0.0,
        }

    precision = cnt["TP"] / cnt["TPFP"]
    recall = cnt["TP"] / cnt["TPFN"]
    f1 = 2 * precision * recall / (precision + recall)
    return {
        f"{prefix}_Prec": precision * 100,
        f"{prefix}_Rec": recall * 100,
        f"{prefix}_F1": f1 * 100,
    }


def classification_micro_f1(outputs, labels, prefix="train"):
    outputs = outputs["classification"] >= 0.5
    labels = labels["classification"]

    cnt = {}
    cnt["TP"] = torch.logical_and(outputs, labels).sum().item()
    cnt["TPFP"] = outputs.sum().item()
    cnt["TPFN"] = labels.sum().item()

    scores = calc_f1(cnt, prefix=prefix)
    return scores, scores[f"{prefix}_F1"]

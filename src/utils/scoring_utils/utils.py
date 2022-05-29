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

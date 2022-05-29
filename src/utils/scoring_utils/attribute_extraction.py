from collections import Counter
from utils.scoring_utils import calc_f1


def tag_to_offset(tags):
    offset = []
    start = None
    for i, tag in enumerate(tags):
        if tag < 0:
            continue
        if tag == 1:
            if start is not None:
                offset.append((start, i))
            start = i
        elif tag == 0 and start is not None:
            offset.append((start, i))
            start = None
    return offset


def attribute_extraction_micro_f1(outputs, labels, all_active_attr_flags, prefix="train"):
    outputs["attribute_extraction"] = outputs["attribute_extraction"].numpy()
    labels["attribute_extraction"] = labels["attribute_extraction"].numpy()

    cnt = Counter()
    for outputs, labels, category in zip(
        outputs["attribute_extraction"], labels["attribute_extraction"], labels["category"]
    ):
        act_outputs = outputs[all_active_attr_flags[category]].tolist()
        act_labels = labels[all_active_attr_flags[category]].tolist()
        for act_output, act_label in zip(act_outputs, act_labels):
            res_offset = tag_to_offset(act_output)
            ans_offset = tag_to_offset(act_label)

            cnt["TP"] += len(set(ans_offset) & set(res_offset))
            cnt["TPFP"] += len(res_offset)
            cnt["TPFN"] += len(ans_offset)

    scores = calc_f1(cnt, prefix=prefix)
    return scores, scores[f"{prefix}_F1"]

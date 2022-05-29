import os
import re

from multiprocessing import Pool
import multiprocessing as multi

from collections import defaultdict

import tqdm

from utils.data_utils import DataUtils
from utils.array_utils import flatten

from utils.tokenization_utils import JanomeBpeTokenizer


def mp_preprocess(inputs):
    category, data, model_dir, output_dir = inputs

    tokenizer = JanomeBpeTokenizer(os.path.join(model_dir, "codecs.txt"))

    offset_error_count = defaultdict(lambda: {"all": 0, "error": 0})
    for d in data:
        d["category"] = category
        d["tokens"] = tokenizer.tokenize(d["text"])

        ## オフセット確認用にBPRの後続記号を取り除いたトークンを用意する
        clean_token = lambda token: token[:-2] if token[-2:] == "@@" else token
        clean_tokens = lambda tokens: list(map(clean_token, tokens))
        cleaned_tokens = list(map(clean_tokens, d["tokens"]))

        token_offsets, front_tokens = [], [0]
        for tokens in cleaned_tokens:
            token_offsets.append([0])
            for token in tokens:
                token_offsets[-1].append(token_offsets[-1][-1] + len(token))
            front_tokens.append(front_tokens[-1] + len(tokens))

        d["tokens"] = flatten(d["tokens"])
        cleaned_tokens = flatten(cleaned_tokens)
        del d["text"]

        # オフセットを各トークンにマップ
        for ann in d["annotation"]:
            offset = ann["text_offset"]
            try:
                s_id, s_offset = offset["start"]["line_id"], offset["start"]["offset"]
                e_id, e_offset = offset["end"]["line_id"], offset["end"]["offset"]
            except KeyError:
                print(ann)
                continue

            ann["token_offset"] = {
                "start": len(token_offsets[s_id]) + front_tokens[s_id],
                "end": len(token_offsets[e_id]) + front_tokens[e_id],
            }
            for i, offset in enumerate(token_offsets[s_id]):
                if offset == s_offset:
                    ann["token_offset"]["start"] = i + front_tokens[s_id]
                    break
                elif offset > s_offset:
                    ann["token_offset"]["start"] = i - 1 + front_tokens[s_id]
                    break

            for i, offset in enumerate(token_offsets[e_id]):
                if offset >= e_offset:
                    ann["token_offset"]["end"] = i + front_tokens[e_id]
                    break

            span_tokens = cleaned_tokens[ann["token_offset"]["start"] : ann["token_offset"]["end"]]
            if ann["text_offset"]["text"] != "".join(span_tokens).replace("▁", " "):
                offset_error_count[ann["attribute"]]["error"] += 1
            offset_error_count[ann["attribute"]]["all"] += 1

            del ann["text_offset"]
            del ann["html_offset"]

    for attr, cnt in offset_error_count.items():
        if cnt["error"] == 0:
            continue
        error_rate = cnt["error"] / cnt["all"] * 100
        print(f"Offset match error: {category}|{attr}|{error_rate:.1f}%")

    DataUtils.JsonL.save(os.path.join(output_dir, f"{category}.json"), data)


def preprocess(cfg):
    annotation_data, attributes = DataUtils.AttrExtData.load(
        os.path.join(cfg.data.dir, cfg.data.annotation_dir)
    )

    output_dir = os.path.join(cfg.data.output_dir, f"{cfg.data.annotation_dir}_prep")
    annotation_output_dir = os.path.join(output_dir, "data")
    os.makedirs(annotation_output_dir, exist_ok=True)

    DataUtils.Json.save(os.path.join(output_dir, "attributes.json"), attributes)

    tasks = [
        (categ, data, cfg.model.dir, annotation_output_dir)
        for categ, data in annotation_data.items()
    ]
    with Pool(multi.cpu_count()) as p, tqdm.tqdm(desc="Preprocessing", total=len(tasks)) as t:
        for _ in p.imap_unordered(mp_preprocess, tasks):
            t.update()

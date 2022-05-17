import os
import re

from multiprocessing import Pool
import multiprocessing as multi

from collections import defaultdict

import tqdm

from janome.tokenizer import Tokenizer

from utils.data_utils import DataUtils
from utils.array_utils import flatten


def mp_preprocess(inputs):
    category, data, output_dir = inputs

    tokenizer = Tokenizer(wakati=True)
    offset_error_count = defaultdict(lambda: {"all": 0, "error": 0})
    for d in data:
        lines = d["text"].splitlines()
        d["tokens"] = []

        token_offsets, front_tokens = [], [0]
        for line in lines:
            tokens = []
            if len(line.strip()) != 0:
                tokens = list(tokenizer.tokenize(line))

                # janomeは先頭のスペースが消えてしまうため、オフセットがずれないよう調節
                m = re.match("^\s*", line)
                front_spaces = [] if m is None else list(m.group(0))
                tokens = front_spaces + tokens
                d["tokens"].append(tokens)

            token_offsets.append([0])
            for token in tokens:
                token_offsets[-1].append(token_offsets[-1][-1] + len(token))
            front_tokens.append(front_tokens[-1] + len(tokens))

        d["tokens"] = flatten(d["tokens"])
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

            span_tokens = d["tokens"][ann["token_offset"]["start"] : ann["token_offset"]["end"]]
            if ann["text_offset"]["text"] != "".join(span_tokens):
                # print(ann["text_offset"]["text"], span_tokens)
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

    tasks = [(categ, data, annotation_output_dir) for categ, data in annotation_data.items()]
    with Pool(multi.cpu_count()) as p, tqdm.tqdm(desc="Preprocessing", total=len(tasks)) as t:
        for _ in p.imap_unordered(mp_preprocess, tasks):
            t.update()

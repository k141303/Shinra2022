import os
import re

from multiprocessing import Pool
import multiprocessing as multi

import tqdm

from utils.tokenization_utils import JanomeBpeTokenizer

from utils.ene_utils import EneData
from utils.data_utils import DataUtils
from utils.array_utils import flatten


def mp_preprocess(inputs):
    i, data, model_dir, num_tokens, output_dir = inputs
    tokenizer = JanomeBpeTokenizer(os.path.join(model_dir, "codecs.txt"))

    for d in data:
        lines = re.findall("[^。]+。?", d["text"])
        tokens = tokenizer.tokenize(lines, max_tokens=num_tokens - 2)
        tokens = flatten(tokens)
        d["tokens"] = tokens[: num_tokens - 2]
        del d["text"]

    DataUtils.JsonL.save(os.path.join(output_dir, f"{i}.json"), data)


def preprocess(cfg):
    output_dir = os.path.join(cfg.data.output_dir, f"{cfg.data.cirrus_name}_prep")
    cirrus_output_dir = os.path.join(output_dir, "data")
    os.makedirs(cirrus_output_dir, exist_ok=True)

    ene_data, pageids = None, None
    if cfg.data.ene_name is not None:
        ene_data = EneData(os.path.join(cfg.data.dir, cfg.data.ene_name))
        ene_data.save_ene_id_list(output_dir)
        pageids = ene_data.get_pageids()

    cirrus_data = DataUtils.CirrusSearch.load(
        os.path.join(cfg.data.dir, cfg.data.cirrus_name),
        pageids=pageids,
        debug_mode=cfg.debug_mode,
    )

    if ene_data is not None:
        for d in cirrus_data:
            d["ENEs"] = ene_data.get_ene_ids(d["pageid"])

    tasks = [
        (
            int(i / 5000),
            cirrus_data[i : i + 5000],
            cfg.model.dir,
            cfg.num_tokens,
            cirrus_output_dir,
        )
        for i in range(0, len(cirrus_data), 5000)
    ]
    with Pool(multi.cpu_count()) as p, tqdm.tqdm(desc="Preprocessing", total=len(tasks)) as t:
        for _ in p.imap_unordered(mp_preprocess, tasks):
            t.update()

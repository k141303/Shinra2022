import os

from multiprocessing import Pool
import multiprocessing as multi

import tqdm
import hydra

from transformers import AutoTokenizer

from utils.ene_utils import EneData
from utils.data_utils import DataUtils


def mp_preprocess(inputs):
    i, data, tokenizer_cls, num_tokens, output_dir = inputs
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_cls)

    for d in data:
        tokens = tokenizer.tokenize(d["text"])[:num_tokens - 2]
        d["token_ids"] = tokenizer.convert_tokens_to_ids(tokens)
        del d["text"]

    DataUtils.JsonL.save(os.path.join(output_dir, f"{i}.json"), data)


def preprocess(cfg):
    cfg.data.dir = hydra.utils.to_absolute_path(cfg.data.dir)
    cfg.data.output_dir = hydra.utils.to_absolute_path(cfg.data.output_dir)

    tokenizer_cls = cfg.tokenizer_cls.replace('/', '_')
    output_dir = os.path.join(cfg.data.output_dir, f"{cfg.data.old_cirrus_name}_{tokenizer_cls}")
    os.makedirs(output_dir, exist_ok=True)

    ene_data = EneData(os.path.join(cfg.data.dir, cfg.data.ene_name))
    ene_data.save_ene_id_list(output_dir)

    cirrus_data = DataUtils.CirrusSearch.load(
        os.path.join(cfg.data.dir, cfg.data.old_cirrus_name),
        pageids=ene_data.get_pageids(),
        debug_mode=cfg.debug_mode
    )

    for d in cirrus_data:
        d["ENEs"] = ene_data.get_ene_ids(d["pageid"])

    cirrus_output_dir = os.path.join(output_dir, "data")
    tasks = [
        (int(i / 5000), cirrus_data[i:i + 5000], cfg.tokenizer_cls, cfg.num_tokens, cirrus_output_dir)
        for i in range(0, len(cirrus_data), 5000)
    ]
    with Pool(multi.cpu_count()) as p, tqdm.tqdm(desc="Preprocessing", total=len(tasks)) as t:
        for _ in p.imap(mp_preprocess, tasks):
            t.update()

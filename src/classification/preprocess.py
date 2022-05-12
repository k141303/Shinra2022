import os

from multiprocessing import Pool
import multiprocessing as multi

import tqdm

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
    ene_data = EneData(os.path.join(cfg.data.dir, cfg.data.ene_name))

    cirrus_data = DataUtils.CirrusSearch.load(
        os.path.join(cfg.data.dir, cfg.data.old_cirrus_name),
        pageids=ene_data.get_pageids(),
        debug_mode=cfg.debug_mode
    )

    for d in cirrus_data:
        d["ENEs"] = ene_data.get_ene_ids(d["pageid"])

    output_dir = os.path.join(cfg.data.output_dir, f"{cfg.data.old_cirrus_name}_{cfg.preprocess.tokenizer_cls}")
    os.makedirs(output_dir, exist_ok=True)
    tasks = [
        (int(i / 5000), cirrus_data[i:i + 5000], cfg.preprocess.tokenizer_cls, cfg.preprocess.num_tokens, output_dir)
        for i in range(0, len(cirrus_data), 5000)
    ]
    with Pool(multi.cpu_count()) as p, tqdm.tqdm(desc="Preprocessing", total=len(tasks)) as t:
        for _ in p.imap(mp_preprocess, tasks):
            t.update()

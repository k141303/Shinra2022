import os
import glob
import random

from multiprocessing import Pool
import multiprocessing as multi

import tqdm

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

from utils.data_utils import DataUtils
from utils.array_utils import padding
from utils.scoring_utils import classification_micro_f1


class ClassificationDataset(Dataset):
    def __init__(self, data, ene_id_list, vocab, num_tokens=512, target_slots=None):
        self.num_tokens = num_tokens
        self.data = data
        self.vocab = vocab
        self.ene_id_list = ene_id_list
        self.num_labels = len(ene_id_list)
        self.target_slots = target_slots

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]

        tokens = ["<s>"] + d["tokens"] + ["</s>"]
        attention_mask = [1] * len(tokens)

        tokens = padding(tokens, "<pad>", self.num_tokens)
        attention_mask = padding(attention_mask, 0, self.num_tokens)

        token_ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]

        item = {
            "pageid": d["pageid"],
            "input_ids": torch.LongTensor(token_ids),
            "attention_mask": torch.LongTensor(attention_mask),
        }

        if d.get("ENEs") is not None:
            ene_indexes = [self.ene_id_list.index(ene_id) for ene_id in d["ENEs"]]
            labels = [i in ene_indexes for i in range(self.num_labels)]
            item["labels"] = torch.LongTensor(labels)

        return item

    def evaluation(self, outputs, labels, prefix="train"):
        return classification_micro_f1(outputs, labels, prefix=prefix)

    @classmethod
    def load_dataset(cls, file_dir, model_dir, num_tokens=512, dev_size=1000, debug_mode=False):
        vocab = DataUtils.Json.load(os.path.join(model_dir, "vocab.json"))
        DataUtils.Json.save("vocab.json", vocab)

        ene_id_list = DataUtils.Json.load(os.path.join(file_dir, "ene_id_list.json"))

        file_paths = glob.glob(os.path.join(file_dir, "data/*.json"))
        assert file_paths, "glob error."
        file_paths.sort()

        if debug_mode:
            file_paths = file_paths[:5]

        data = []
        with Pool(multi.cpu_count()) as p, tqdm.tqdm(desc="Loading", total=len(file_paths)) as t:
            for _data in p.imap(DataUtils.JsonL.load, file_paths):
                data += _data
                t.update()

        all_indexes = list(range(len(data)))
        random.shuffle(all_indexes)
        train_indexes = sorted(all_indexes[dev_size:])
        dev_indexes = sorted(all_indexes[:dev_size])

        train_data = [data[i] for i in train_indexes]
        dev_data = [data[i] for i in dev_indexes]

        train_dataset = cls(train_data, ene_id_list, vocab, num_tokens=num_tokens)
        dev_dataset = cls(dev_data, ene_id_list, vocab, num_tokens=num_tokens)

        return train_dataset, dev_dataset

    @classmethod
    def load_pred_dataset(
        cls, file_dir, ene_id_list, target_path, num_tokens=512, debug_mode=False
    ):
        file_paths = glob.glob(os.path.join(file_dir, "data/*.json"))
        assert file_paths, "glob error."
        file_paths.sort()

        if debug_mode:
            file_paths = file_paths[:5]

        target_slots = {}
        for d in DataUtils.JsonL.load(target_path):
            if "page_id" in d:
                target_slots[d["page_id"]] = d
            else:
                target_slots[d["pageid"]] = d

        data = []
        with Pool(multi.cpu_count()) as p, tqdm.tqdm(desc="Loading", total=len(file_paths)) as t:
            for _data in p.imap(DataUtils.JsonL.load, file_paths):
                for d in _data:
                    if str(d["pageid"]) not in target_slots:
                        continue
                    data.append(d)
                t.update()

        vocab = DataUtils.Json.load("vocab.json")
        return cls(data, ene_id_list, vocab, num_tokens=num_tokens, target_slots=target_slots)

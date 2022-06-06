import os
import re
import glob
import random

from multiprocessing import Pool
import multiprocessing as multi
from tokenize import group

import numpy as np

import tqdm

import torch
from torch.utils.data import Dataset

from utils.data_utils import DataUtils
from utils.array_utils import flatten, padding, slide
from utils.scoring_utils import attribute_extraction_micro_f1


class AttributeExtractionDataset(Dataset):
    def __init__(self, data, attributes, vocab, num_tokens=512, duplicate_tokens=64):
        self.attributes = {categ: sorted(attrs) for categ, attrs in attributes.items()}
        self.all_attributes = sorted(set(flatten(attributes.values())))
        self.num_labels = len(self.all_attributes)
        self.attr2index = dict(zip(self.all_attributes, range(len(self.all_attributes))))

        self.vocab = vocab
        self.num_tokens = num_tokens
        self.duplicate_tokens = duplicate_tokens

        self.data, self.all_active_attr_flags = self.make_input_data(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        category, pageid, inputs = self.data[i]

        tokens, pos_labels = map(list, zip(*inputs))
        tokens = ["<s>"] + tokens + ["</s>"]

        pos_labels = [[-100] * len(self.attributes[category])] + pos_labels
        attention_mask = [1] * len(tokens)

        tokens = padding(tokens, "<pad>", self.num_tokens)
        pos_labels = padding(pos_labels, [-100] * len(self.attributes[category]), self.num_tokens)
        attention_mask = padding(attention_mask, 0, self.num_tokens)

        token_ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]

        labels = torch.ones((self.num_tokens, len(self.all_attributes)), dtype=torch.long) * -100
        pos_labels = torch.LongTensor(pos_labels)
        for i, attr in enumerate(self.attributes[category]):
            j = self.attr2index[attr]
            labels[:, j] = pos_labels[:, i]

            """
            zeros = torch.zeros_like(labels[:, j])
            pos = torch.where(labels[:, j] >= 0, labels[:, j], zeros)
            id2token = {v: k for k, v in self.vocab.items()}
            for token_id, _pos in zip(token_ids, pos):
                print(attr, _pos.item(), id2token[token_id])

            pass
            """

        return {
            "category": category,
            "pageid": pageid,
            "input_ids": torch.LongTensor(token_ids),
            "attention_mask": torch.LongTensor(attention_mask),
            "labels": torch.LongTensor(labels),
        }

    def make_input_data(self, orig_data):
        data = []
        for d in tqdm.tqdm(orig_data, desc="Making input data"):
            labels = {attr: [0] * len(d["tokens"]) for attr in self.attributes[d["category"]]}
            for ann in d["annotation"]:
                labels[ann["attribute"]][ann["token_offset"]["start"]] = 1
                for i in range(ann["token_offset"]["start"] + 1, ann["token_offset"]["end"]):
                    labels[ann["attribute"]][i] = 2

            # ignore_labels = [-100] * len(d["tokens"])
            labels = [labels[attr] for attr in self.attributes[d["category"]]]
            labels = [*zip(*labels)]
            w_inputs = slide(
                list(zip(d["tokens"], labels)),
                window=self.num_tokens - 2,
                dup=self.duplicate_tokens,
            )
            data += [(d["category"], d["pageid"], inputs) for inputs in w_inputs]

        all_active_attr_flags = {}
        for categ, attrs in self.attributes.items():
            all_active_attr_flags[categ] = [attr in attrs for attr in self.all_attributes]
        return data, all_active_attr_flags

    def evaluation(self, outputs, labels, prefix="train"):
        return attribute_extraction_micro_f1(
            outputs, labels, self.all_active_attr_flags, prefix=prefix
        )

    @classmethod
    def load_dataset(
        cls,
        file_dir,
        model_dir,
        num_tokens=512,
        duplicate_tokens=64,
        dev_size=20,
        debug_mode=False,
    ):
        vocab = DataUtils.Json.load(os.path.join(model_dir, "vocab.json"))

        attributes = DataUtils.Json.load(os.path.join(file_dir, "attributes.json"))

        file_paths = glob.glob(os.path.join(file_dir, "data/*.json"))
        assert file_paths, "glob error."
        file_paths.sort()

        if debug_mode:
            file_paths = [file_path for file_path in file_paths if "Person" in file_path]
            # file_paths = file_paths[:2]

        all_data = []
        with Pool(min(multi.cpu_count(), len(file_paths))) as p, tqdm.tqdm(
            desc="Loading", total=len(file_paths)
        ) as t:
            for data in p.imap(DataUtils.JsonL.load, file_paths):
                all_data.append(data)
                t.update()

        ext_category_name = lambda x: re.match(".*/(.*?).json", x).group(1)
        categories = list(map(ext_category_name, file_paths))
        attributes = {category: attributes[category] for category in categories}

        train_data, dev_data = [], []
        for data in all_data:
            indexes = list(range(len(data)))
            random.shuffle(indexes)

            train_indexes = indexes[dev_size:]
            dev_indexes = indexes[:dev_size]

            train_data += [data[i] for i in train_indexes]
            dev_data += [data[i] for i in dev_indexes]

        train_dataset = cls(
            train_data, attributes, vocab, num_tokens=num_tokens, duplicate_tokens=duplicate_tokens
        )
        dev_dataset = cls(
            dev_data, attributes, vocab, num_tokens=num_tokens, duplicate_tokens=duplicate_tokens
        )

        return train_dataset, dev_dataset

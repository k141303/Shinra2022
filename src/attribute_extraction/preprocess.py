import os
import re

from transformers import AutoTokenizer
from transformers.models.bert.tokenization_bert import (
    WordpieceTokenizer,
    whitespace_tokenize,
)
from transformers.models.bert_japanese.tokenization_bert_japanese import BertJapaneseTokenizer

from utils.data_utils import DataUtils


class MyWordpieceTokenizer(WordpieceTokenizer):
    def tokenize(self, text):
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class MyBertJapaneseTokenizer(BertJapaneseTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.do_subword_tokenize:
            if self.subword_tokenizer_type == "wordpiece":
                self.subword_tokenizer = MyWordpieceTokenizer(
                    vocab=self.vocab, unk_token=self.unk_token
                )


def mp_preprocess(inputs):
    category, data, tokenizer_cls, output_dir = inputs

    tokenizer = MyBertJapaneseTokenizer.from_pretrained(tokenizer_cls)
    for d in data:
        lines = d["text"].splitlines()
        for line in lines:
            if len(line) == 0:
                continue
            tokens = tokenizer.tokenize(line)
            remove_header = lambda x: re.sub("^##", "", x)
            tokens = list(map(remove_header, tokens))
            _line = "".join(tokens)
            if len(re.sub("\s", "", line)) != len(_line):
                print(line, _line)
        pass
    pass


def preprocess(cfg):
    annotation_data, attributes = DataUtils.AttrExtData.load(
        os.path.join(cfg.data.dir, cfg.data.annotation_dir)
    )

    tokenizer_cls = cfg.tokenizer_cls.replace("/", "_")
    output_dir = os.path.join(cfg.data.output_dir, f"{cfg.data.annotation_dir}_{tokenizer_cls}")
    annotation_output_dir = os.path.join(output_dir, "data")
    os.makedirs(annotation_output_dir, exist_ok=True)

    DataUtils.Json.save(os.path.join(output_dir, "attributes.json"), attributes)

    tasks = [
        (categ, data, cfg.tokenizer_cls, annotation_output_dir)
        for categ, data in annotation_data.items()
    ]
    list(map(mp_preprocess, tasks))
    pass

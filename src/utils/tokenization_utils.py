import os
import re

import codecs

from janome.tokenizer import Tokenizer as JanomeTokenizer
from subword_nmt.apply_bpe import BPE


class BpeTokenizer:
    def __init__(self, codes, separator="@@"):
        self.separator = separator
        codes = codecs.open(codes, encoding="utf-8")
        self.bpe = BPE(codes, separator=separator)

    def tokenize(self, text):
        if len(text) == 0:
            return []
        encoded = self.bpe.process_line(text)
        return encoded.split(" ")


class JanomeBpeTokenizer(object):
    def __init__(self, bpe_codes_path):
        self.janome = JanomeTokenizer(wakati=True)
        self.bpe = None
        if os.path.exists(bpe_codes_path):
            self.bpe = BpeTokenizer(bpe_codes_path)

    def tokenize(self, text):
        sentences = []
        for line in text.split("\n"):
            tokens = self.janome.tokenize(line)

            l_space = re.match("^(\s*)", line).group(1)
            tokens = [l_space] + list(tokens)

            tokens = [re.sub("\s", " ▁ ", t) for t in tokens]

            if self.bpe is not None:
                tokens = self.bpe.tokenize(" ".join(tokens))
            else:
                tokens = " ".join(tokens).split(" ")
            tokens = list(filter(len, tokens))
            sentences.append(tokens)
        return sentences

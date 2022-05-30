import os

import torch
from torch import nn
from torch.nn import functional as F

from transformers import RobertaConfig, RobertaModel

from utils.data_utils import DataUtils


class ClassificationModel(nn.Module):
    def __init__(self, bert_dir, label_list, num_tokens=512, load_bert=True):
        super().__init__()

        self.label_list = label_list
        self.num_labels = len(label_list)
        self.num_tokens = num_tokens

        self.config = RobertaConfig.from_pretrained(os.path.join(bert_dir, "config.json"))
        if load_bert:
            self.bert = RobertaModel.from_pretrained(
                os.path.join(bert_dir, "pytorch_model.bin"), config=self.config
            )
        else:
            self.bert = RobertaModel(self.config)

        self.linear = nn.Linear(self.config.hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        embeddings = outputs[0][:, 0]
        logits = self.linear(embeddings)

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.view(-1).float())
            return (
                {"classification": F.sigmoid(logits)},
                {"classification": loss},
                {"classification": labels},
            )

        return {"classification": F.sigmoid(logits)}

    def save(self):
        config = {
            "label_list": self.label_list,
            "num_tokens": self.num_tokens,
        }
        DataUtils.Json.save("config.json", config)
        torch.save(self.state_dict(), "pytorch_model.bin")

    @classmethod
    def load(cls):
        config = DataUtils.Json.load("config.json")
        state_dict = torch.load("pytorch_model.bin")
        model = cls(config["bert_cls"], config["label_list"], num_tokens=config["num_tokens"])
        model.load_state_dict(state_dict)
        return model

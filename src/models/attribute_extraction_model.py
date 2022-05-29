import os
import torch
from torch import nn

from transformers import RobertaConfig, RobertaModel

from utils.array_utils import flatten
from utils.data_utils import DataUtils


class AttributeExtractionModel(nn.Module):
    def __init__(
        self, bert_dir, attributes, load_bert=True, num_tokens=512, weights=[1.0, 100, 100]
    ):
        super().__init__()

        self.attributes = attributes
        self.all_attributes = sorted(set(flatten(attributes.values())))
        self.num_labels = len(self.all_attributes)
        self.num_tokens = num_tokens
        self.weights = weights

        self.config = RobertaConfig.from_pretrained(os.path.join(bert_dir, "config.json"))
        if load_bert:
            self.bert = RobertaModel.from_pretrained(
                os.path.join(bert_dir, "pytorch_model.bin"), config=self.config
            )
        else:
            self.bert = RobertaModel(self.config)

        self.linear = nn.Linear(self.config.hidden_size, self.num_labels * 3)

    def forward(self, input_ids, attention_mask=None, labels=None):

        outputs = self.bert(input_ids, attention_mask=attention_mask)
        embeddings = outputs[0]
        logits = self.linear(embeddings)

        logits = logits.view(input_ids.size(0), -1, self.num_labels, 3)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(self.weights, device=labels.device))
            loss = loss_fct(logits.view(-1, 3), labels.view(-1))

            return (
                {"attribute_extraction": logits.argmax(-1).transpose(-1, -2)},
                {"attribute_extraction": loss},
                {"attribute_extraction": labels.transpose(-1, -2)},
            )

        return {"attribute_extraction": logits.argmax(-1).transpose(-1, -2)}

    def save(self):
        config = {
            "attributes": self.attributes,
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

import tqdm
import mlflow

import torch

from trainers import BaseTrainer
from utils.array_utils import decompose_array_tensors


class AttributeExtractionTrainer(BaseTrainer):
    def evaluation(self):
        self.model.eval()
        all_outputs, all_labels = [], []
        for batch in tqdm.tqdm(self.dev_dataloader, desc="Evaluating"):
            with torch.no_grad():
                outputs = self.model(**self.make_forward_inputs(batch, with_labels=False))
            all_outputs.append({k: v.cpu() for k, v in outputs.items()})
            all_labels.append({"attribute_extraction": batch["labels"].transpose(-1, -2)})
            all_labels.append({"category": batch["category"]})

        all_outputs = decompose_array_tensors(all_outputs)
        all_labels = decompose_array_tensors(all_labels)

        scores, score = self.dev_dataset.evaluation(all_outputs, all_labels, prefix="dev")
        mlflow.log_metrics(scores, self.updates)
        self.model.train()

        return score

    def forward_and_backward_step(self, batch):
        self.update_labels.append({"category": batch["category"]})
        super().forward_and_backward_step(batch)

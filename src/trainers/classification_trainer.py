import tqdm
import mlflow

import torch

from trainers import BaseTrainer
from utils.array_utils import decompose_array_tensors


class ClassificationTrainer(BaseTrainer):
    def evaluation(self):
        self.model.eval()
        all_outputs, all_labels = [], []
        for batch in tqdm.tqdm(self.dev_dataloader, desc="Evaluating"):
            with torch.no_grad():
                outputs = self.model(**self.make_forward_inputs(batch, with_labels=False))
            all_outputs.append({k: v.cpu() for k, v in outputs.items()})
            all_labels.append({"classification": batch["labels"]})

        all_outputs = decompose_array_tensors(all_outputs)
        all_labels = decompose_array_tensors(all_labels)

        scores, score = self.dev_dataset.evaluation(all_outputs, all_labels, prefix="dev")
        mlflow.log_metrics(scores, self.updates)
        self.model.train()

        return score

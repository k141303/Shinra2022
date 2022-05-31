import os
import tqdm

import torch

from predictors import BasePredictor
from trainers import ClassificationTrainer

from utils.array_utils import decompose_array_tensors
from utils.data_utils import DataUtils


class ClassificationPredictor(BasePredictor, ClassificationTrainer):
    def predict(self):
        all_outputs, all_pageids = [], []
        for batch in tqdm.tqdm(self.dataloader, desc="Predicting"):
            with torch.no_grad():
                outputs = self.model(**self.make_forward_inputs(batch, with_labels=False))
            all_outputs.append({k: v.cpu() for k, v in outputs.items()})
            all_pageids += batch["pageid"]

        all_outputs = decompose_array_tensors(all_outputs)
        all_outputs = all_outputs["classification"]

        is_pos = lambda x: x[1]
        data = []
        for pageid, outputs in tqdm.tqdm(
            zip(all_pageids, all_outputs), total=len(all_pageids), desc="Converting labels"
        ):
            bin_outputs = (outputs >= 0.5).tolist()
            pos_probs = outputs[outputs >= 0.5].tolist()
            pos_outputs = list(filter(is_pos, zip(self.dataset.ene_id_list, bin_outputs)))
            if len(pos_outputs) >= 1:
                ene_ids, _ = zip(*pos_outputs)
            else:
                ene_ids = [self.dataset.ene_id_list[torch.argmax(outputs).item()]]
                pos_probs = [torch.max(outputs).item()]

            formated_outputs = []
            for ene_id, prob in zip(ene_ids, pos_probs):
                formated_outputs.append(
                    {
                        "ENE": ene_id,
                        "prob": prob,
                    }
                )
            self.dataset.target_slots[pageid]["ENEs"][self.cfg.data.ene_tag] = formated_outputs

        predictions = list(self.dataset.target_slots.values())
        num_completion = sum([bool(d["ENEs"]) for d in predictions])
        completion_rate = num_completion / len(predictions) * 100 if len(predictions) else -1
        print(f"Target Completion Rate:{completion_rate:.2f}%")

        os.makedirs("predictions", exist_ok=True)
        output_path = os.path.join("predictions", self.cfg.data.target_name)
        DataUtils.JsonL.save(output_path, predictions)

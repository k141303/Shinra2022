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
            pos_outputs = list(filter(is_pos, zip(self.dataset.ene_id_list, bin_outputs)))
            if len(pos_outputs) >= 1:
                ene_ids, _ = zip(*pos_outputs)
            else:
                ene_ids = [self.dataset.ene_id_list[torch.argmax(outputs).item()]]
            data.append({"pageid": pageid, "ENEs": ene_ids})
        DataUtils.JsonL.save("predicts.json", data)

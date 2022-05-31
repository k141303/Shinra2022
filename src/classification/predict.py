import os

from models import ClassificationModel
from datasets import ClassificationDataset
from predictors import ClassificationPredictor


def predict(cfg):
    model = ClassificationModel.load()
    dataset = ClassificationDataset.load_pred_dataset(
        os.path.join(cfg.data.dir, cfg.data.dataset_name),
        model.label_list,
        num_tokens=model.num_tokens,
        debug_mode=cfg.debug_mode,
        target_path=os.path.join(cfg.data.dir, cfg.data.target_name),
    )
    predictor = ClassificationPredictor(cfg, model, dataset)
    predictor.predict()

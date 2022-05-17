import os

from models import ClassificationModel
from datasets import ClassificationDataset
from predictors import ClassificationPredictor


def predict(cfg):
    model = ClassificationModel.load()
    dataset = ClassificationDataset.load_pred_dataset(
        os.path.join(cfg.data.dir, cfg.data.dataset_dir),
        model.bert_cls,
        model.label_list,
        num_tokens=model.num_tokens,
        debug_mode=cfg.debug_mode,
    )
    predictor = ClassificationPredictor(cfg, model, dataset)
    predictor.predict()

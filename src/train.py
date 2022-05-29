import hydra
import torch

import mlflow

from transformers import set_seed

from classification.train import train as classification_train
from attribute_extraction.train import train as attribute_extraction_train


@hydra.main(config_path="../config", config_name="train")
def main(cfg):
    mlflow.start_run()

    cfg = cfg.train
    cfg.data.dir = hydra.utils.to_absolute_path(cfg.data.dir)
    cfg.model.bert.dir = hydra.utils.to_absolute_path(cfg.model.bert.dir)

    cfg.device.n_gpu = torch.cuda.device_count()
    cfg.device.device = "cuda" if torch.cuda.is_available() and not cfg.device.no_cuda else "cpu"
    set_seed(cfg.seed)

    if cfg.name == "classification":
        classification_train(cfg)
    if cfg.name == "attribute_extraction":
        attribute_extraction_train(cfg)

    mlflow.end_run()


if __name__ == "__main__":
    main()

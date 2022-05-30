import os

from datasets import ClassificationDataset
from models import ClassificationModel
from trainers import ClassificationTrainer


def train(cfg):
    train_dataset, dev_dataset = ClassificationDataset.load_dataset(
        os.path.join(cfg.data.dir, cfg.data.dataset_dir),
        cfg.model.bert.dir,
        num_tokens=cfg.model.num_tokens,
        dev_size=cfg.data.dev_size,
        debug_mode=cfg.debug_mode,
    )
    model = ClassificationModel(
        cfg.model.bert.dir, train_dataset.ene_id_list, num_tokens=cfg.model.num_tokens
    )
    trainer = ClassificationTrainer(cfg, model, train_dataset, dev_dataset)

    trainer.train()

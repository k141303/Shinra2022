import os

from datasets import AttributeExtractionDataset
from models import AttributeExtractionModel
from trainers import AttributeExtractionTrainer


def train(cfg):
    train_dataset, dev_dataset = AttributeExtractionDataset.load_dataset(
        os.path.join(cfg.data.dir, cfg.data.dataset_dir),
        cfg.model.bert.dir,
        num_tokens=cfg.model.num_tokens,
        duplicate_tokens=cfg.data.duplicate_tokens,
        dev_size=cfg.data.dev_size,
        debug_mode=cfg.debug_mode,
    )
    model = AttributeExtractionModel(
        cfg.model.bert.dir, train_dataset.attributes, num_tokens=cfg.model.num_tokens
    )
    trainer = AttributeExtractionTrainer(cfg, model, train_dataset, dev_dataset)

    trainer.train()

import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from utils.harf_precision_utils import set_half_precision


class BasePredictor(object):
    def __init__(self, cfg, model, dataset):
        self.cfg = cfg
        self.model = model
        self.dataset = dataset

        self.model.to(cfg.device.device)
        self.model, self.amp = set_half_precision(cfg, self.model)

        if cfg.device.n_gpu != 1:
            self.model = DataParallel(self.model)

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.cfg.dataloader.predict.batch_size,
            num_workers=self.cfg.dataloader.num_workers,
        )
        self.model.eval()

    def predict(self):
        raise NotImplementedError()

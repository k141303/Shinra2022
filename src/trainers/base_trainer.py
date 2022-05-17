import os

import random
from collections import defaultdict
from copy import deepcopy

import tqdm
import mlflow

import numpy as np

import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from utils.optim_utils import make_optimizer_and_scheduler
from utils.harf_precision_utils import set_half_precision
from utils.array_utils import decompose_array_tensors


class BaseTrainer(object):
    def __init__(self, cfg, model, train_dataset, dev_dataset):
        self.cfg = cfg
        self.model = model
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset

        self.train_indexes = self.make_train_indexes()

        self.optimizer, self.scheduler = make_optimizer_and_scheduler(cfg, self.model)

        self.model.to(cfg.device.device)
        self.model, self.optimizer, self.amp = set_half_precision(cfg, self.model, self.optimizer)

        if cfg.device.n_gpu != 1:
            self.model = DataParallel(self.model)

        self.steps = 0
        self.updates = 0
        self.update_outputs = []
        self.update_labels = []
        self.update_losses = defaultdict(float)
        self.best_eval_score = -1

        self.load_checkpoint()

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.dataloader.train.batch_size,
            num_workers=self.cfg.dataloader.num_workers,
            sampler=self.train_indexes,
        )
        self.dev_dataloader = DataLoader(
            self.dev_dataset,
            batch_size=self.cfg.dataloader.eval.batch_size,
            num_workers=self.cfg.dataloader.num_workers,
        )

    def make_train_indexes(self):
        total_indexes = (
            self.cfg.gradient_accumulation_steps
            * self.cfg.total_updates
            * self.cfg.dataloader.train.batch_size
        )
        indexes = list(range(len(self.train_dataset)))
        train_indexes = []
        while len(train_indexes) < total_indexes:
            if self.cfg.dataloader.train.shuffle:
                random.shuffle(indexes)
            train_indexes += deepcopy(indexes)
        return train_indexes[:total_indexes]

    def train(self):
        self.model.train()

        self.p_bar = tqdm.tqdm(total=self.cfg.total_updates, desc="Training:epoch:  0.0%")
        self.p_bar.update(self.updates)

        for batch in self.train_dataloader:
            self.forward_and_backward_step(batch)

            if self.steps % self.cfg.gradient_accumulation_steps == 0:
                self.update_model_parameters()  # self.updates += 1

                if self.updates % self.cfg.log_updates == 0:
                    self.update_log()

                if self.updates % self.cfg.eval_updates == 0:
                    eval_score = self.evaluation()

                    if self.best_eval_score is None or eval_score > self.best_eval_score:
                        self.best_eval_score = eval_score
                        self.save_model()

                if self.updates % self.cfg.checkpoint_updates == 0:
                    self.save_checkpoint()

        self.p_bar.close()

    def mean_loss(self, loss):
        mean_losses = {}
        for key, value in loss.items():
            if self.cfg.device.n_gpu > 1:
                value = torch.mean(value)
            mean_losses[key] = torch.mean(value)

        mean_loss = torch.mean(torch.stack(list(mean_losses.values())))
        return mean_loss, mean_losses

    def make_forward_inputs(self):
        NotImplementedError()

    def forward_and_backward_step(self, batch):
        outputs, losses = self.model(**self.make_forward_inputs(batch))

        loss, losses = self.mean_loss(losses)

        if self.cfg.device.fp16:
            with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        for key, each_loss in losses.items():
            self.update_losses["loss_" + key] += each_loss.item() / (
                self.cfg.gradient_accumulation_steps * self.cfg.log_updates
            )

        self.update_outputs.append({key: value.cpu() for key, value in outputs.items()})
        self.update_labels.append({"classification": batch["labels"]})

        self.steps += 1

    def update_model_parameters(self):
        self.optimizer.step()
        self.scheduler.step()
        self.model.zero_grad()
        self.optimizer.zero_grad()

        self.updates += 1
        self.p_bar.update()
        epoch = (
            (
                self.cfg.gradient_accumulation_steps
                * self.cfg.dataloader.train.batch_size
                * self.updates
            )
            / len(self.train_dataset)
            * 100
        )
        self.p_bar.set_description(f"Training:epoch:{epoch:>5.1f}%")

    def evaluation(self):
        raise NotImplementedError()

    def update_log(self):
        mlflow.log_metrics(self.update_losses, self.updates)
        self.update_losses = defaultdict(int)

        update_outputs = decompose_array_tensors(self.update_outputs)
        update_labels = decompose_array_tensors(self.update_labels)

        eval_scores, _ = self.train_dataset.evaluation(update_outputs, update_labels)
        mlflow.log_metrics(eval_scores, self.updates)
        self.update_outputs = []
        self.update_labels = []

        return update_outputs, update_labels

    def make_checkpoint(self):
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        checkpoint = {
            "steps": self.steps,
            "updates": self.updates,
            "best_eval_score": self.best_eval_score,
            "model": model_to_save.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "amp": self.amp.state_dict(),
            "random": random.getstate(),
            "torch": torch.get_rng_state(),
            "torch_random": torch.random.get_rng_state(),
            "cuda_random": torch.cuda.get_rng_state(),
            "cuda_random_all": torch.cuda.get_rng_state_all(),
            "np_random": np.random.get_state(),
        }
        return checkpoint

    def save_checkpoint(self):
        checkpoint = self.make_checkpoint()
        torch.save(checkpoint, "checkpoint.bin")

    def load_checkpoint(self):
        if not self.cfg.use_checkpoint or not os.path.exists("checkpoint.bin"):
            return
        checkpoint = torch.load("checkpoint.bin")
        self.steps = checkpoint["steps"]
        self.updates = checkpoint["updates"]
        self.best_eval_score = checkpoint["best_eval_score"]
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        if self.amp is not None:
            self.amp.load_state_dict(checkpoint["amp"])
        random.setstate(checkpoint["random"])
        torch.set_rng_state(checkpoint["torch"])
        torch.random.set_rng_state(checkpoint["torch_random"])
        torch.cuda.set_rng_state(checkpoint["cuda_random"])
        try:
            torch.cuda.torch.cuda.set_rng_state_all(checkpoint["cuda_random_all"])
        except:
            pass
        np.random.set_state(checkpoint["np_random"])
        self.train_indexes = self.train_indexes[self.steps * self.cfg.dataloader.train.batch_size :]

    def save_model(self):
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save()

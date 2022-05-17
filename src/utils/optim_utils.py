from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)


def select_optimizer(cfg, model):
    if cfg.optim.name == "adamw":
        return AdamW(
            model.parameters(),
            lr=cfg.optim.lr,
            betas=(cfg.optim.adam_B1, cfg.optim.adam_B2),
            weight_decay=cfg.optim.weight_decay,
            eps=cfg.optim.adam_eps,
        )
    raise NotImplementedError(f"オプティマイザー{cfg.optim.name}は未実装です。")


def select_scheduler(cfg, optimizer):
    if cfg.scheduler.name == "linear_with_warmup":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.scheduler.warmup_updates,
            num_training_steps=cfg.total_updates,
        )
    raise NotImplementedError(f"スケジューラー{cfg.scheduler.name}は未実装です。")


def make_optimizer_and_scheduler(cfg, model):
    optimizer = select_optimizer(cfg, model)
    scheduler = select_scheduler(cfg, optimizer)
    return optimizer, scheduler

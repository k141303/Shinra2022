import torch


def set_half_precision(cfg, model, optimizer=None):
    if not cfg.device.fp16 or cfg.device.device == "cpu":
        if optimizer is None:
            return model, None
        return model, optimizer, None

    try:
        import apex
        from apex import amp
    except ModuleNotFoundError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
        )
    amp.register_float_function(torch, "sigmoid")
    if optimizer is None:
        model = apex.amp.initialize(model, opt_level=cfg.device.fp16_opt_level)
        return model, amp
    else:
        model, optimizer = apex.amp.initialize(
            model, optimizer, opt_level=cfg.device.fp16_opt_level
        )

    return model, optimizer, amp

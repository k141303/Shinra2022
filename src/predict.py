import hydra

import torch

from classification.predict import predict as classification_predict


@hydra.main(config_path="../config", config_name="predict")
def main(cfg):
    cfg = cfg.predict
    cfg.data.dir = hydra.utils.to_absolute_path(cfg.data.dir)

    cfg.device.n_gpu = torch.cuda.device_count()
    cfg.device.device = "cuda" if torch.cuda.is_available() and not cfg.device.no_cuda else "cpu"

    if cfg.name == "classification":
        classification_predict(cfg)


if __name__ == "__main__":
    main()

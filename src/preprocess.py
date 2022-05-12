import hydra

from classification.preprocess import preprocess as classification_preprocess


@hydra.main(config_path="../config", config_name="preprocess")
def main(cfg):
    cfg.data.dir = hydra.utils.to_absolute_path(cfg.data.dir)
    cfg.data.output_dir = hydra.utils.to_absolute_path(cfg.data.output_dir)

    if cfg.preprocess.name == "classification":
        classification_preprocess(cfg)


if __name__ == "__main__":
    main()

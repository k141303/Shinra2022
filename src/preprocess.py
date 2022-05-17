import hydra

from classification.preprocess import preprocess as classification_preprocess


@hydra.main(config_path="../config", config_name="preprocess")
def main(cfg):
    if cfg.preprocess.name == "classification":
        classification_preprocess(cfg.preprocess)


if __name__ == "__main__":
    main()

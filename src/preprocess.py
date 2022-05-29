import hydra

from classification.preprocess import preprocess as classification_preprocess
from attribute_extraction.preprocess import preprocess as attribute_extraction_preprocess


@hydra.main(config_path="../config", config_name="preprocess")
def main(cfg):
    cfg = cfg.preprocess
    cfg.data.dir = hydra.utils.to_absolute_path(cfg.data.dir)
    cfg.model.dir = hydra.utils.to_absolute_path(cfg.model.dir)
    cfg.data.output_dir = hydra.utils.to_absolute_path(cfg.data.output_dir)

    if cfg.name == "classification":
        classification_preprocess(cfg)
    if cfg.name == "attribute_extraction":
        attribute_extraction_preprocess(cfg)


if __name__ == "__main__":
    main()

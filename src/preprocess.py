import hydra

from classification.preprocess import preprocess as classification_preprocess
from attribute_extraction.preprocess import preprocess as attribute_extraction_preprocess


@hydra.main(config_path="../config", config_name="preprocess")
def main(cfg):
    cfg.preprocess.data.dir = hydra.utils.to_absolute_path(cfg.preprocess.data.dir)
    cfg.preprocess.data.output_dir = hydra.utils.to_absolute_path(cfg.preprocess.data.output_dir)

    if cfg.preprocess.name == "classification":
        classification_preprocess(cfg.preprocess)
    if cfg.preprocess.name == "attribute_extraction":
        attribute_extraction_preprocess(cfg.preprocess)


if __name__ == "__main__":
    main()

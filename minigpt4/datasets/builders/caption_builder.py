from minigpt4.common.registry import registry
from .base_dataset_builder import BaseDatasetBuilder
from minigpt4.datasets.datasets.coco_caption_dataset import COCOCapDataset, COCOCapEvalDataset # noqa


@registry.register_builder("coco_caption")
class COCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapDataset
    eval_dataset_cls = COCOCapEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_cap.yaml",
    }
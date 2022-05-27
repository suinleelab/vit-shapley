from vit_shapley.datamodules.base_datamodule import BaseDataModule
from vit_shapley.datamodules.datasets.ImageNette_dataset import ImageNetteDataset


class ImageNetteDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return ImageNetteDataset

    @property
    def dataset_name(self):
        return "ImageNette"


if __name__ == '__main__':
    pass

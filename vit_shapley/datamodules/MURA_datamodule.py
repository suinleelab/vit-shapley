from vit_shapley.datamodules.base_datamodule import BaseDataModule
from vit_shapley.datamodules.datasets.MURA_dataset import MURADataset


class MURADataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return MURADataset

    @property
    def dataset_name(self):
        return "MURA"


if __name__ == '__main__':
    pass

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class BaseDataModule(LightningDataModule):
    def __init__(self, dataset_location,
                 transforms_train, transforms_val, transforms_test,
                 explanation_location_train, explanation_mask_amount_train, explanation_mask_ascending_train,
                 explanation_location_val, explanation_mask_amount_val, explanation_mask_ascending_val,
                 explanation_location_test, explanation_mask_amount_test, explanation_mask_ascending_test,
                 num_workers, per_gpu_batch_size, test_data_split):
        super().__init__()

        self.dataset_location = dataset_location

        self.transforms_train = transforms_train
        self.transforms_val = transforms_val
        self.transforms_test = transforms_test
        self.test_data_split = test_data_split

        self.explanation_location_train = explanation_location_train
        self.explanation_mask_amount_train = explanation_mask_amount_train
        self.explanation_mask_ascending_train = explanation_mask_ascending_train

        self.explanation_location_val = explanation_location_val
        self.explanation_mask_amount_val = explanation_mask_amount_val
        self.explanation_mask_ascending_val = explanation_mask_ascending_val

        self.explanation_location_test = explanation_location_test
        self.explanation_mask_amount_test = explanation_mask_amount_test
        self.explanation_mask_ascending_test = explanation_mask_ascending_test

        self.num_workers = num_workers
        self.batch_size = per_gpu_batch_size

        self.setup_flag = False

    @property
    def dataset_cls(self):
        raise NotImplementedError("return tuple of dataset class")

    @property
    def dataset_name(self):
        raise NotImplementedError("return name of dataset")

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(
            dataset_location=self.dataset_location,
            transform_params=self.transforms_train,
            explanation_location=self.explanation_location_train,
            explanation_mask_amount=self.explanation_mask_amount_train,
            explanation_mask_ascending=self.explanation_mask_ascending_train,
            split="train",
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(
            dataset_location=self.dataset_location,
            transform_params=self.transforms_val,
            explanation_location=self.explanation_location_val,
            explanation_mask_amount=self.explanation_mask_amount_val,
            explanation_mask_ascending=self.explanation_mask_ascending_val,
            split="val",
        )

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(
            dataset_location=self.dataset_location,
            transform_params=self.transforms_test,
            explanation_location=self.explanation_location_test,
            explanation_mask_amount=self.explanation_mask_amount_test,
            explanation_mask_ascending=self.explanation_mask_ascending_test,
            split=self.test_data_split,
        )

    def setup(self, stage):
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()

            self.setup_flag = True

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            prefetch_factor=2,
            persistent_workers=True,
            drop_last=True,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            prefetch_factor=2,
            persistent_workers=True,
            drop_last=True,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            prefetch_factor=2,
            persistent_workers=False,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loader

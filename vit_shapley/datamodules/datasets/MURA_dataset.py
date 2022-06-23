import os
import random

import pandas as pd
import sklearn.model_selection
from torch.utils.data import DataLoader

from vit_shapley.datamodules.datasets.base_dataset import BaseDataset


class MURADataset(BaseDataset):
    def __init__(self, dataset_location, transform_params, explanation_location, explanation_mask_amount,
                 explanation_mask_ascending, split='train'):
        # use imagenet mean,std for normalization
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        super().__init__(transform_params=transform_params, explanation_location=explanation_location,
                         explanation_mask_amount=explanation_mask_amount,
                         explanation_mask_ascending=explanation_mask_ascending, img_channels=3)

        # https://github.com/pengshiqi/MURA/blob/master/dataset/dataset.py
        self.dataset_location = dataset_location

        self.split = split

        self.labels = [
            'negative',  # normal
            'positive'  # abnormal
        ]
        self.data = self.get_data_list()

    def get_data_list(self):
        # Load files containing labels, and perform train/valid split if necessary
        print('')
        if self.split == 'train' or self.split == 'val':
            data = pd.read_csv(os.path.join(self.dataset_location, 'train_image_paths.csv'), header=None,
                               names=["path"])
            data["split"] = data["path"].map(lambda x: x.split("/")[1])
            assert (data["split"] == "train").all()
        elif self.split == 'test':
            data = pd.read_csv(os.path.join(self.dataset_location, 'valid_image_paths.csv'), header=None,
                               names=["path"])
            data["split"] = data["path"].map(lambda x: x.split("/")[1])
            assert (data["split"] == "valid").all()
        else:
            raise ValueError("Invalid fold: {:s}".format(str(self.split)))

        data["body_part"] = data["path"].map(lambda x: x.split("/")[2])
        assert data["body_part"].str.startswith("XR").all()

        data["patient"] = data["path"].map(lambda x: x.split("/")[3])
        assert data["patient"].str.startswith("patient").all()

        data["study"] = data["path"].map(lambda x: x.split("/")[4].split('_')[0])

        data["label"] = data["path"].map(lambda x: x.split("/")[4].split('_')[1])
        assert ((data["label"] == "positive") | (data["label"] == "negative")).all()

        if self.split == 'train' or self.split == 'val':
            idx_train, idx_val = sklearn.model_selection.train_test_split(data["patient"].unique(), random_state=44,
                                                                          test_size=0.1)
            if self.split == 'train':
                data = data[data["patient"].isin(idx_train)]
            else:
                data = data[data["patient"].isin(idx_val)]
        # labels = np.eye(10)[data['noisy_labels_0'].map(lambda x: self.labels.index(x))]
        labels = data['label'].map(lambda x: self.labels.index(x))  # .map(lambda x: self.labels.index(x))
        img_paths = data["path"].map(
            lambda x: os.path.join(self.dataset_location, x.replace("MURA-v1.1/", ""))).values.tolist()
        data_list = [{'img_path': img_path, 'label': [label], 'dataset': self.__class__.__name__}
                     for img_path, label in zip(img_paths, labels)]
        random.Random(42).shuffle(data_list)
        return data_list


if __name__ == '__main__':
    from vit_shapley.config import dataset_MURA_ROAR

    transform_params = {}

    dataset_train = MURADataset(dataset_location="/homes/gws/chanwkim/network_drive/sync/MURA-v1.1/",
                                explanation_location="/homes/gws/chanwkim/ViT_shapley/results/4_1_explanation_generate/MURA/vit_base_patch16_224_ours_train.pickle",
                                explanation_mask_amount=180,
                                explanation_mask_ascending=True,
                                transform_params=dataset_MURA_ROAR()["transforms_train"],
                                split='train')
    dataloader_train = DataLoader(dataset_train, batch_size=16)
    next(iter(dataloader_train))

    transform_params = {}

    dataset_train = MURADataset(dataset_location="/homes/gws/chanwkim/ViT_shapley/data/MURA-v1.1/",
                                explanation_location=None,
                                explanation_mask_amount=None,
                                explanation_mask_ascending=None,
                                transform_params=transform_params, split='train')
    dataset_valid = MURADataset(dataset_location="/homes/gws/chanwkim/ViT_shapley/data/MURA-v1.1/",
                                explanation_location=None,
                                explanation_mask_amount=None,
                                explanation_mask_ascending=None,
                                transform_params=transform_params, split='val')
    dataset_test = MURADataset(dataset_location="/homes/gws/chanwkim/ViT_shapley/data/MURA-v1.1/",
                               explanation_location=None,
                               explanation_mask_amount=None,
                               explanation_mask_ascending=None,
                               transform_params=transform_params, split='test')
    pd.Series([i['label'] for i in dataset_train.data]).value_counts()
    [dataset_train.df[dataset_train.df["patient"] == study]["body_part"] for study in
     dataset_train.df["patient"].unique()]
    # dataset_valid = MURADataset(dataset_location=None, transform_params=transform_params, split='val')
    # dataset_test = MURADataset(dataset_location=None, transform_params=transform_params, split='test')
    """
    dataset_train: 33071//64
    dataset_valid: 3737//64
    dataset_test: 3197//64
    """

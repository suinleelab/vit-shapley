import numpy as np
import pandas as pd
import random
import sklearn
import sklearn.model_selection
from fastai.vision.all import untar_data, URLs
from torch.utils.data import DataLoader

from vit_shapley.datamodules.datasets.base_dataset import BaseDataset


class ImageNetteDataset(BaseDataset):
    def __init__(self, dataset_location, transform_params, explanation_location, explanation_mask_amount,
                 explanation_mask_ascending, split='train'):
        # use imagenet mean,std for normalization
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        super().__init__(transform_params=transform_params, explanation_location=explanation_location,
                         explanation_mask_amount=explanation_mask_amount,
                         explanation_mask_ascending=explanation_mask_ascending, img_channels=3)

        # self.dataset_location = dataset_location
        # import torchvision
        self.dataset_location = dataset_location

        self.split = split

        self.labels = [
            'n02979186',
            'n03417042',
            'n01440764',
            'n02102040',
            'n03028079',
            'n03888257',
            'n03394916',
            'n03000684',
            'n03445777',
            'n03425413']
        self.data = self.get_data_list()

    def get_data_list(self):
        path = untar_data(URLs.IMAGENETTE_160)
        # Load files containing labels, and perform train/valid split if necessary
        data = pd.read_csv(path.joinpath('noisy_imagenette.csv'))
        if self.split == 'train':
            data = data[~data['is_valid']]
        elif self.split == 'val' or self.split == 'test':
            data = data[data['is_valid']]
            idx_val, idx_test = sklearn.model_selection.train_test_split(data.index, random_state=44, test_size=0.5)
            if self.split == 'val':
                data = data.loc[idx_val]
            else:
                data = data.loc[idx_test]

        else:
            raise ValueError("Invalid fold: {:s}".format(str(self.split)))

        # labels = np.eye(10)[data['noisy_labels_0'].map(lambda x: self.labels.index(x))]
        labels = data['noisy_labels_0'].map(lambda x: self.labels.index(x))
        img_paths = data['path'].map(lambda x: str(path.joinpath(x))).values.tolist()
        data_list = [{'img_path': img_path, 'label': label, 'dataset': self.__class__.__name__}
                     for img_path, label in zip(img_paths, labels)]
        random.Random(42).shuffle(data_list)
        return data_list


if __name__ == '__main__':
    from vit_shapley.config import dataset_ImageNette

    transform_params = {}

    dataset_train = ImageNetteDataset(dataset_location=None,
                                      explanation_location="/homes/gws/chanwkim/ViT_shapley/results/4_1_explanation_generate/ImageNette/vit_base_patch16_224_ours_train.pickle",
                                      explanation_mask_amount=180,
                                      explanation_mask_ascending=True,
                                      transform_params=dataset_ImageNette()["transforms_train"],
                                      split='train')
    dataloader_train = DataLoader(dataset_train, batch_size=16)
    next(iter(dataloader_train))

    dataset_valid = ImageNetteDataset(dataset_location=None, transform_params=transform_params, split='val')
    dataset_test = ImageNetteDataset(dataset_location=None, transform_params=transform_params, split='test')
    """
    dataset_train: 9469//64
    dataset_valid: 1962//64
    dataset_test: 1963//64
    """

import numpy as np
import pandas as pd
import random
import sklearn
import sklearn.model_selection
import os
from torch.utils.data import DataLoader

from vit_shapley.datamodules.datasets.base_dataset import BaseDataset


class PetDataset(BaseDataset):
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

        self.labels = ['Abyssinian',
                       'american_bulldog',
                       'american_pit_bull_terrier',
                       'basset_hound',
                       'beagle',
                       'Bengal',
                       'Birman',
                       'Bombay',
                       'boxer',
                       'British_Shorthair',
                       'chihuahua',
                       'Egyptian_Mau',
                       'english_cocker_spaniel',
                       'english_setter',
                       'german_shorthaired',
                       'great_pyrenees',
                       'havanese',
                       'japanese_chin',
                       'keeshond',
                       'leonberger',
                       'Maine_Coon',
                       'miniature_pinscher',
                       'newfoundland',
                       'Persian',
                       'pomeranian',
                       'pug',
                       'Ragdoll',
                       'Russian_Blue',
                       'saint_bernard',
                       'samoyed',
                       'scottish_terrier',
                       'shiba_inu',
                       'Siamese',
                       'Sphynx',
                       'staffordshire_bull_terrier',
                       'wheaten_terrier',
                       'yorkshire_terrier']
        self.data = self.get_data_list()

    def get_data_list(self):
        # Load files containing labels, and perform train/valid split if necessary
        data=pd.read_csv(os.path.join(self.dataset_location, "annotations/list.txt"), sep=' ', skiprows=[0,1,2,3,4,5], names=["classid","species","breed"])
        idx_train, idx_valtest = sklearn.model_selection.train_test_split(data.index, random_state=44, test_size=0.2)
        idx_val, idx_test = sklearn.model_selection.train_test_split(idx_valtest, random_state=44, test_size=0.5)

        if self.split == 'train':
            data = data[data.index.isin(idx_train)]
        elif self.split == 'val':
            data = data[data.index.isin(idx_val)]
        elif self.split == 'test':
            data = data[data.index.isin(idx_test)]
        else:
            raise ValueError("Invalid fold: {:s}".format(str(self.split)))

        # labels = np.eye(10)[data['noisy_labels_0'].map(lambda x: self.labels.index(x))]
        #labels = data["classid"].astype(int)-1 #data['noisy_labels_0'].map(lambda x: self.labels.index(x))
        labels = data.index.map(lambda x: "_".join(x.split('_')[:-1])).map(lambda x: self.labels.index(x))

        img_paths = data.index.map(lambda x: str(os.path.join(os.path.join(self.dataset_location, "images/"),x))+".jpg").values.tolist()
        data_list = [{'img_path': img_path, 'label': label, 'dataset': self.__class__.__name__}
                     for img_path, label in zip(img_paths, labels)]
        random.Random(42).shuffle(data_list)
        return data_list


if __name__ == '__main__':
    from vit_shapley.config import dataset_Pet

    transform_params = {}

    dataset_train = PetDataset(dataset_location="/homes/gws/chanwkim/network_drive/sync/pet_dataset",
                               explanation_location=None,
                               explanation_mask_amount=None,
                               explanation_mask_ascending=None,
                              transform_params=dataset_Pet()["transforms_train"],
                              split='train')
    dataset_val = PetDataset(dataset_location="/homes/gws/chanwkim/network_drive/sync/pet_dataset",
                               explanation_location=None,
                               explanation_mask_amount=None,
                               explanation_mask_ascending=None,
                               transform_params=dataset_Pet()["transforms_val"],
                               split='val')
    dataset_test = PetDataset(dataset_location="/homes/gws/chanwkim/network_drive/sync/pet_dataset",
                               explanation_location=None,
                               explanation_mask_amount=None,
                               explanation_mask_ascending=None,
                               transform_params=dataset_Pet()["transforms_test"],
                               split='test')

    dataloader_train = DataLoader(dataset_train, batch_size=16)
    next(iter(dataloader_train))

    dataset_valid = PetDataset(dataset_location=None, transform_params=transform_params, split='val')
    dataset_test = PetDataset(dataset_location=None, transform_params=transform_params, split='test')
    """
    dataset_train: 9469//64
    dataset_valid: 1962//64
    dataset_test: 1963//64
    """

import json
import numpy as np
import os
import pickle
import random
import torch
from PIL import Image, ImageFilter
from copy import deepcopy
from torchvision.transforms import RandomApply, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, \
    CenterCrop, Resize, RandomCrop, RandomGrayscale, RandomAffine, RandomPerspective, \
    Normalize, RandomErasing, Compose, ColorJitter, ToTensor, RandomResizedCrop

try:
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import RandomSolarize
except:
    pass


class RandomGaussianBlur:
    # copy-past from https://github.com/facebookresearch/dino
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() < self.prob:
            return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max)))
        else:
            return img


class AugmentSimpleRepeat:
    def __init__(self, num_augmentations):
        self.num_augmentations = num_augmentations

    def __call__(self, image, transform):
        return [transform(image) for _ in range(self.num_augmentations)]


class AugmentMultiCrop:
    def __init__(self, num_repeats_list, size_list, scale_list):
        self.n_crops = num_repeats_list
        self.sizes = size_list
        self.scales = scale_list
        self.crop_transform_list = []

        assert len(num_repeats_list) == len(size_list)
        assert len(num_repeats_list) == len(scale_list)

        for i, num_repeats in enumerate(num_repeats_list):
            for _ in range(num_repeats):
                self.crop_transform_list.append(RandomResizedCrop(size=size_list[i], scale=scale_list[i],
                                                                  interpolation=InterpolationMode.BICUBIC))

    def __call__(self, image, transform):
        return [transform(crop_transform(image)) for crop_transform in self.crop_transform_list]


def save_json(data, filename):
    filename = os.path.abspath(filename)
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'w') as wfile:
        json.dump(data, wfile)


def load_json(filename):
    filename = os.path.abspath(filename)
    with open(filename, "r") as rfile:
        data = json.load(rfile)
    return data


def pil_loader(img_path, n_channels):
    with open(img_path, 'rb') as f:
        img = Image.open(f)
        if n_channels == 3:
            return img.convert('RGB')
        elif n_channels == 1:
            return img.convert('L')
        elif n_channels == 4:
            return img.convert('RGBA')
        else:
            raise NotImplementedError("PIL only supports 1,3 and 4 channel inputs. Use cv2 instead")


def adapt_path(path_original, dict_keys):
    path_list = ['l0.cs.washington.edu', 'l1lambda.cs.washington.edu', 'l2lambda.cs.washington.edu',
                 'l3.cs.washington.edu', 'deeper.cs.washington.edu', 'sync']

    dict_keys = list(dict_keys)

    for path1 in path_list:
        if path1 in path_original:
            for path2 in path_list:
                path_replaced = path_original.replace(path1, path2)
                if path_replaced in dict_keys:
                    return path_replaced
    raise ValueError(f"not found {path_original}")


def get_relative_value(x, random_seed=None):
    assert len(x.shape) == 1

    if isinstance(random_seed, int):
        rng = np.random.default_rng(random_seed)
        perm = rng.permutation(np.arange(len(x)))
    else:
        perm = np.random.permutation(np.arange(len(x)))

    argsorted = np.arange(len(x))[perm][np.argsort(x[perm])]
    relative_value = np.argsort(argsorted)

    return relative_value


class BaseDataset(torch.utils.data.Dataset):
    """Base dataset class that actual datasets, e.g. Cifar10, subclasses.

    This class only has torchvision.transforms for augmentation.
    Not intended to be used directly.
    """

    def __init__(self, transform_params, explanation_location, explanation_mask_amount, explanation_mask_ascending,
                 img_channels):

        assert (
                       explanation_location is None and explanation_mask_amount is None and explanation_mask_ascending is None) or (
                       explanation_location is not None and explanation_mask_amount is not None and explanation_mask_ascending is not None), f"explanation_location: {explanation_location}, explanation_mask_amount: {explanation_mask_amount}, explanation_mask_ascending: {explanation_mask_ascending}"

        self.img_channels = img_channels
        self.transform_params = transform_params
        self.explanation_location = explanation_location
        self.explanation_mask_amount = explanation_mask_amount
        self.explanation_mask_ascending = explanation_mask_ascending

        self.transforms, self.transform_resize, self.augmentation_function = self.parse_transforms_params(
            self.transform_params)

        self.data = None

        if self.explanation_location is None:
            self.explanation_loaded = None
        else:
            with open(self.explanation_location, 'rb') as f:
                self.explanation_loaded = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        label = torch.as_tensor(self.data[idx]['label'])

        img_path = self.data[idx]['img_path']
        img_path_png = '.'.join(img_path.split('.')[:-1]) + '.png'
        if os.path.exists(img_path_png):
            img = pil_loader(img_path_png, self.img_channels)
        else:
            img = pil_loader(img_path, self.img_channels)

        # resize

        if self.transform_resize is not None:
            img = self.transform_resize(img)

        # transform with augmentation
        img = self.augmentation_function(img, self.transforms)

        img = img[0] if len(img) == 1 and isinstance(img, list) else img

        if self.explanation_mask_amount is not None:
            assert img.shape == (3, 224, 224)
            if len(self.labels) > 2:
                explanation = self.explanation_loaded[img_path]['explanation']
                # explanation = explanation + np.random.RandomState(42).uniform(low=0, high=1e-40, size=explanation.shape)
                if len(explanation.shape) == 1:
                    pass
                elif len(explanation.shape) == 2:
                    explanation = explanation[label.item()]
                else:
                    raise
                explanation = get_relative_value(explanation)

                if self.explanation_mask_ascending:
                    explanation_mask = explanation > (np.sort(explanation)[::][
                                                          self.explanation_mask_amount - 1] if self.explanation_mask_amount != 0 else -np.inf)
                else:
                    explanation_mask = explanation < (np.sort(explanation)[::-1][
                                                          self.explanation_mask_amount - 1] if self.explanation_mask_amount != 0 else np.inf)
            elif len(self.labels) == 2:
                explanation = self.explanation_loaded[adapt_path(img_path, self.explanation_loaded.keys())][
                    'explanation']
                # explanation = explanation + np.random.RandomState(42).uniform(low=0, high=1e-40, size=explanation.shape)
                if len(explanation.shape) == 1:
                    pass
                elif len(explanation.shape) == 2:
                    explanation = explanation[0]
                else:
                    raise
                explanation = get_relative_value(explanation)

                if label.item() == 1:
                    if self.explanation_mask_ascending:
                        explanation_mask = explanation > (np.sort(explanation)[::][
                                                              self.explanation_mask_amount - 1] if self.explanation_mask_amount != 0 else -np.inf)
                    else:
                        explanation_mask = explanation < (np.sort(explanation)[::-1][
                                                              self.explanation_mask_amount - 1] if self.explanation_mask_amount != 0 else np.inf)
                elif label.item() == 0:
                    if self.explanation_mask_ascending:
                        explanation_mask = explanation < (np.sort(explanation)[::-1][
                                                              self.explanation_mask_amount - 1] if self.explanation_mask_amount != 0 else np.inf)
                    else:
                        explanation_mask = explanation > (np.sort(explanation)[::][
                                                              self.explanation_mask_amount - 1] if self.explanation_mask_amount != 0 else -np.inf)
                else:
                    raise
                # print(explanation_mask)
            else:
                raise

            # print(explanation_mask, self.explanation_mask_amount)
            explanation_mask = explanation_mask.reshape(-1, 14, 14)
            explanation_mask = torch.Tensor(explanation_mask)
            explanation_mask = torch.repeat_interleave(torch.repeat_interleave(explanation_mask, 16, dim=2), 16, dim=1)
            img = img * explanation_mask
            # print(img)
            # sdsds

        return {"images": img, "labels": label, "path": img_path}

    def parse_transforms_params(self, transforms_params):
        transforms_params_copied = deepcopy(transforms_params)

        """
        transform_resize
        """
        transform_resize = None
        if "Resize" in transforms_params:
            if transforms_params['Resize']['apply']:
                if ('height' in transforms_params['Resize'] and 'width' in transforms_params['Resize']) and \
                        ('size' not in transforms_params['Resize']):
                    transform_resize = Resize((transforms_params['Resize']['height'],
                                               transforms_params['Resize']['width']))
                if ('height' not in transforms_params['Resize'] and 'width' not in transforms_params['Resize']) and \
                        ('size' in transforms_params['Resize']):
                    transform_resize = Resize(transforms_params['Resize']['size'])
            del transforms_params_copied['Resize']

        """
        augmentation_function
        """
        augmentation_function = AugmentSimpleRepeat(num_augmentations=1)
        if "AugmentMultiCrop" in transforms_params and "Augment_SimpleRepeat" in transforms_params:
            assert not (transforms_params['Augment_SimpleRepeat']['apply'] and transforms_params['AugmentMultiCrop'][
                'apply']), "SimpleRepeat and MultiCrop cannot be activated at the same time."

        if "AugmentMultiCrop" in transforms_params:
            if transforms_params['AugmentMultiCrop']['apply']:
                augmentation_function = AugmentMultiCrop(
                    num_repeats_list=transforms_params["AugmentMultiCrop"]["n_crops"],
                    size_list=transforms_params["AugmentMultiCrop"]["sizes"],
                    scale_list=transforms_params["AugmentMultiCrop"]["scales"])
            del transforms_params_copied['AugmentMultiCrop']

        if "AugmentSimpleRepeat" in transforms_params:
            if transforms_params['AugmentSimpleRepeat']['apply']:
                augmentation_function = AugmentSimpleRepeat(
                    num_augmentations=transforms_params['AugmentSimpleRepeat']['n_repeats'])
            del transforms_params_copied['AugmentSimpleRepeat']

        """
        transforms_list
        https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
        """
        transforms_list = []

        if "CenterCrop" in transforms_params_copied:
            if transforms_params_copied['CenterCrop']['apply']:
                transforms_list.append(CenterCrop((transforms_params_copied['CenterCrop']['height'],
                                                   transforms_params_copied['CenterCrop']['width'])))
            del transforms_params_copied['CenterCrop']

        if "RandomCrop" in transforms_params_copied:
            if transforms_params_copied['RandomCrop']['apply']:
                padding = transforms_params_copied['RandomCrop']['padding']
                transforms_list.append(RandomCrop((transforms_params_copied['RandomCrop']['height'],
                                                   transforms_params_copied['RandomCrop']['width']),
                                                  padding=padding if padding > 0 else None))
            del transforms_params_copied['RandomCrop']

        if "RandomResizedCrop" in transforms_params_copied:
            if transforms_params_copied['RandomResizedCrop']['apply']:
                transforms_list.append(RandomResizedCrop(size=transforms_params_copied['RandomResizedCrop']['size'],
                                                         scale=transforms_params_copied['RandomResizedCrop']['scale'],
                                                         interpolation=InterpolationMode.BILINEAR))
            del transforms_params_copied['RandomResizedCrop']

        if "VerticalFlip" in transforms_params_copied:
            if transforms_params_copied['VerticalFlip']['apply']:
                transforms_list.append(RandomVerticalFlip(p=transforms_params_copied['VerticalFlip']['p']))
            del transforms_params_copied['VerticalFlip']

        if "HorizontalFlip" in transforms_params_copied:
            if transforms_params_copied['HorizontalFlip']['apply']:
                transforms_list.append(RandomHorizontalFlip(p=transforms_params_copied['HorizontalFlip']['p']))
            del transforms_params_copied['HorizontalFlip']

        if "RandomRotation" in transforms_params_copied:
            if transforms_params_copied['RandomRotation']['apply']:
                transforms_list.append(
                    RandomApply(
                        torch.nn.ModuleList(
                            [RandomRotation(degrees=transforms_params_copied['RandomRotation']['angle'])]),
                        p=transforms_params_copied['RandomRotation']['p']))
            del transforms_params_copied['RandomRotation']

        if "ColorJitter" in transforms_params_copied:
            if transforms_params_copied['ColorJitter']['apply']:
                temp_d = transforms_params_copied['ColorJitter']
                transforms_list.append(
                    RandomApply(torch.nn.ModuleList([ColorJitter(brightness=temp_d['brightness'],
                                                                 contrast=temp_d['contrast'],
                                                                 saturation=temp_d['saturation'],
                                                                 hue=temp_d['hue'])]), p=temp_d['p']))
            del transforms_params_copied['ColorJitter']

        if "RandomGrayscale" in transforms_params_copied:
            if transforms_params_copied['RandomGrayscale']['apply']:
                transforms_list.append(RandomGrayscale(p=transforms_params_copied['RandomGrayscale']['p']))
            del transforms_params_copied['RandomGrayscale']

        if "RandomGaussianBlur" in transforms_params_copied:
            if transforms_params_copied['RandomGaussianBlur']['apply']:
                transforms_list.append(
                    RandomGaussianBlur(p=transforms_params_copied['RandomGaussianBlur']['p'],
                                       radius_min=transforms_params_copied['RandomGaussianBlur']['radius_min'],
                                       radius_max=transforms_params_copied['RandomGaussianBlur']['radius_max']))
            del transforms_params_copied['RandomGaussianBlur']

        if "RandomAffine" in transforms_params_copied:
            if transforms_params_copied['RandomAffine']['apply']:
                temp_d = transforms_params_copied['RandomAffine']
                transforms_list.append(
                    RandomApply(torch.nn.ModuleList([RandomAffine(degrees=temp_d['degrees'],
                                                                  translate=temp_d['translate'],
                                                                  scale=temp_d['scale'],
                                                                  shear=temp_d['shear'])]),
                                p=temp_d['p']))
            del transforms_params_copied['RandomAffine']

        if "RandomPerspective" in transforms_params_copied:
            if transforms_params_copied['RandomPerspective']['apply']:
                transforms_list.append(
                    RandomPerspective(transforms_params_copied['RandomPerspective']['distortion_scale'],
                                      p=transforms_params_copied['RandomPerspective']['p']))
            del transforms_params_copied['RandomPerspective']
        if "RandomSolarize" in transforms_params_copied:
            if transforms_params_copied['RandomSolarize']['apply']:
                transforms_list.append(RandomSolarize(threshold=transforms_params_copied['RandomSolarize']['threshold'],
                                                      p=transforms_params_copied['RandomSolarize']['p']))
            del transforms_params_copied['RandomSolarize']

        transforms_list.append(ToTensor())

        if "Normalize" in transforms_params_copied:
            if transforms_params_copied['Normalize']['apply']:
                transforms_list.append(Normalize(mean=self.mean,
                                                 std=self.std))
            del transforms_params_copied['Normalize']
        if "RandomErasing" in transforms_params_copied:
            if transforms_params_copied['RandomErasing']['apply']:
                temp_d = transforms_params_copied['RandomErasing']
                transforms_list.append(RandomErasing(scale=temp_d['scale'],
                                                     ratio=temp_d['ratio'],
                                                     value=temp_d['value'],
                                                     p=temp_d['p']))
            del transforms_params_copied['RandomErasing']

        """
        ignored params check
        """
        assert len(transforms_params_copied) == 0, f'Not supported {transforms_params_copied}'

        return Compose(transforms_list), transform_resize, augmentation_function

    @staticmethod
    def get_validation_ids(total_size=None, val_size=None, json_path=None, seed_n=42):
        """ Gets the total size of the dataset, and the validation size (as int or float [0,1]
        as well as a json path to save the validation ids and it
        returns: the train / validation split)"""

        if (total_size is not None and val_size is not None) and json_path is None:
            # if val_size is 0~1, convert to real integer value.
            if val_size < 1:
                val_size = int(total_size * val_size)

            total_ids = list(range(total_size))
            random.Random(seed_n).shuffle(total_ids)

            train_split = total_ids[val_size:]
            val_split = total_ids[:val_size]

            json_data = {"train_split": train_split, "val_split": val_split}
            save_json(json_data, json_path)
        if (total_size is None and val_size is None) and json_path is not None:
            json_data = load_json(json_path)

            train_split = json_data["train_split"]
            val_split = json_data["val_split"]
        else:
            raise

        return train_split, val_split

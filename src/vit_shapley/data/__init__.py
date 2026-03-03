import os

from vit_shapley.data.imagenette import (
    download_imagenette,
    get_imagenette_dataset,
    get_imagenette_transforms,
)
from vit_shapley.data.mura import (
    get_mura_dataset,
    get_mura_transforms,
)
from vit_shapley.data.pet import (
    download_pet,
    get_pet_dataset,
    get_pet_transforms,
)

_DATASET_REGISTRY = {
    "imagenette": get_imagenette_dataset,
    "mura": get_mura_dataset,
    "pet": get_pet_dataset,
}


def get_dataset(
    name: str,
    root: str | os.PathLike,
    split: str = "train",
    image_size: int = 224,
    transform=None,
    download: bool = False,
):
    """Dispatch to the appropriate dataset loader by name.

    Args:
        name: Dataset name (``"imagenette"``, ``"mura"``, or ``"pet"``).
        root: Root data directory.
        split: ``"train"``, ``"val"``, or ``"test"``.
        image_size: Spatial resolution for default transforms.
        transform: Override the default transforms.
        download: If ``True`` download the dataset if absent.

    Raises:
        ValueError: If *name* is not a recognised dataset.
    """
    if name not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {sorted(_DATASET_REGISTRY.keys())}"
        )
    return _DATASET_REGISTRY[name](
        root=root,
        split=split,
        image_size=image_size,
        transform=transform,
        download=download,
    )


def resolve_num_classes(dataset, target_type: str) -> int:
    """Validate ``target_type`` against the dataset and return model ``num_classes``.

    Rules:
      - ``"binary"``: dataset must have exactly 2 classes → returns 1
        (single logit, sigmoid).
      - ``"multiclass"``: dataset must have ≥ 3 classes → returns
        ``len(dataset.classes)`` (softmax).

    Args:
        dataset: A dataset object with a ``.classes`` attribute.
        target_type: ``"binary"`` or ``"multiclass"``.

    Raises:
        ValueError: If *target_type* is not recognised, or if the dataset
            class count is inconsistent with the chosen target type.
    """
    n = len(dataset.classes)

    if target_type == "binary":
        if n != 2:
            raise ValueError(
                f"target_type='binary' requires a dataset with exactly 2 classes, "
                f"but got {n} classes: {dataset.classes}"
            )
        return 1
    elif target_type == "multiclass":
        if n < 3:
            raise ValueError(
                f"target_type='multiclass' requires a dataset with 3 or more classes, "
                f"but got {n} classes: {dataset.classes}. "
                f"Use target_type='binary' for 2-class datasets."
            )
        return n
    else:
        raise ValueError(
            f"Unknown target_type '{target_type}'. "
            f"Must be 'binary' or 'multiclass'."
        )


__all__ = [
    "download_imagenette",
    "get_imagenette_transforms",
    "get_imagenette_dataset",
    "get_mura_transforms",
    "get_mura_dataset",
    "download_pet",
    "get_pet_transforms",
    "get_pet_dataset",
    "get_dataset",
    "resolve_num_classes",
]

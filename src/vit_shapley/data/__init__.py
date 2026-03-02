import os

from vit_shapley.data.imagenette import (
    download_imagenette,
    get_imagenette_dataset,
    get_imagenette_transforms,
)
from vit_shapley.data.pet import (
    download_pet,
    get_pet_dataset,
    get_pet_transforms,
)

_DATASET_REGISTRY = {
    "imagenette": get_imagenette_dataset,
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
        name: Dataset name (``"imagenette"`` or ``"pet"``).
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


__all__ = [
    "download_imagenette",
    "get_imagenette_transforms",
    "get_imagenette_dataset",
    "download_pet",
    "get_pet_transforms",
    "get_pet_dataset",
    "get_dataset",
]

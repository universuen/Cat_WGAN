from torchvision import transforms

import context
from pokemon_generator import config
from pokemon_generator.datasets import RealImageDataset


def test_real_image_dataset():
    dataset = RealImageDataset(
        config.path.data,
        transform=transforms.Compose(
            [
                lambda x:x[:3],
                transforms.Resize(config.data.image_size),
                transforms.CenterCrop(config.data.image_size),
                lambda x: x.double(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5)
                )
            ]
        )
    )
    for image, label in dataset:
        assert label == 1
        assert image.shape == (3, config.data.image_size, config.data.image_size)


if __name__ == '__main__':
    test_real_image_dataset()

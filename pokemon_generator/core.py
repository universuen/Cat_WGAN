import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

from pokemon_generator import config
from pokemon_generator.datasets import RealImageDataset
from pokemon_generator import models
from pokemon_generator.logger import Logger


class PokemonGenerator:
    def __init__(
            self,
            model_name: str = None,
    ):
        self.logger = Logger(self.__class__.__name__)
        self.logger.info(f'model path: {config.path.models / model_name}')
        self.model_path = str(config.path.models / model_name)
        self.model = None

    def load_model(self):
        self.model = models.Generator(
            input_size=config.data.latent_vector_size,
            output_size=config.data.image_size,
        )
        self.model.load_state_dict(
            torch.load(self.model_path)
        )
        self.model.eval()
        self.logger.info('model was loaded successfully')

    def save_model(self):
        torch.save(
            self.model.state_dict(),
            self.model_path
        )
        self.logger.info('model was saved successfully')

    def train(
            self,
            plots_dir='',
    ):
        self.logger.info('started training new model')
        # prepare data
        dataset = RealImageDataset(
            config.path.training_data,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    lambda x: x[:3],
                    transforms.Resize(config.data.image_size),
                    transforms.CenterCrop(config.data.image_size),
                    lambda x: x.float(),
                ]
            ),
        )
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
        )

        # prepare models
        def init_weights(layer: nn.Module):
            layer_name = layer.__class__.__name__
            if 'Conv' in layer_name:
                nn.init.normal_(layer.weight.data, 0.0, 0.02)
            elif layer_name == 'BatchNorm2d':
                nn.init.normal_(layer.weight.data, 1.0, 0.02)
                nn.init.constant_(layer.bias.data, 0)

        generator_model = models.Generator(
            input_size=config.data.latent_vector_size,
            output_size=config.data.image_size
        )
        generator_model.apply(init_weights)
        discriminator_model = models.Discriminator(
            input_size=config.data.image_size
        )
        discriminator_model.apply(init_weights)

        # prepare the criterion and optimizers
        criterion = nn.BCELoss()
        generator_optimizer = optim.Adam(
            generator_model.parameters(),
            lr=config.training.learning_rate,
            betas=(0.5, 0.999)
        )
        discriminator_optimizer = optim.Adam(
            discriminator_model.parameters(),
            lr=config.training.learning_rate,
            betas=(0.5, 0.999)
        )

        # train models
        generator_losses = []
        discriminator_losses = []

        # collect a generated result with the fixed latent vector in every epoch
        fixed_lv = torch.randn(
            3,
            config.data.latent_vector_size,
            1,
            1,
            device=config.device
        )
        samples = []

        for epoch in range(config.training.epochs):
            print(f'Epoch: {epoch + 1}')

            for idx, (real_images, labels) in enumerate(data_loader, 0):
                print(f'\rProcess: {(idx + 1) * 100 / len(data_loader): .2f}%', end='')

                discriminator_model.zero_grad()

                # feed the discriminator with real images
                real_images = real_images.to(config.device)
                labels = labels.to(config.device).float()
                y = discriminator_model(real_images)
                loss_real = criterion(y.view(-1), labels)
                loss_real.backward()

                # feed the discriminator with fake images
                noises = torch.randn(
                    config.training.batch_size,
                    config.data.latent_vector_size,
                    1,
                    1,
                    device=config.device
                )
                labels = torch.zeros(config.training.batch_size)
                fake_images = generator_model(noises)
                y = discriminator_model(fake_images.detach())
                loss_fake = criterion(y.view(-1), labels)
                loss_fake.backward()

                # optimize the model and collect the loss
                discriminator_optimizer.step()
                discriminator_losses.append(loss_real.item() + loss_fake.item())

                generator_model.zero_grad()

                # the target for the generator is letting the discriminator thinks its products are real,
                # so the labels should all be 1
                labels = torch.ones(config.training.batch_size)
                y = discriminator_model(fake_images)
                loss = criterion(y.view(-1), labels)
                loss.backward()
                generator_optimizer.step()
                generator_losses.append(loss.item())

            print(
                f"\n"
                f"Discriminator loss: {discriminator_losses[-1]}\n"
                f"Generator loss: {generator_losses[-1]}\n"
            )
            with torch.no_grad():
                fake_images = generator_model(fixed_lv).detach().cpu()
            samples.append(make_grid(fake_images))

        self.model = generator_model
        self.save_model()

        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(generator_losses, label="generator")
        plt.plot(discriminator_losses, label="discriminator")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(fname=plots_dir + 'losses.jpg')

        fig = plt.figure()
        plt.axis("off")
        plots = [
            [
                plt.imshow(
                    np.transpose(
                        img,
                        (1, 2, 0),
                    ),
                    animated=True,
                )
            ]
            for img in samples
        ]
        ani = animation.ArtistAnimation(
            fig=fig,
            artists=plots,
            interval=30,
            blit=True
        )
        ani.save(filename=plots_dir + 'animation.gif')

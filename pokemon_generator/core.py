import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np

from . import config
from .datasets import RealImageDataset
from . import models
from .logger import Logger
from ._utils import init_weights, train_d_model, train_g_model, save_samples, show_samples, denormalize


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
        ).to(config.device)
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

    def generate(self, seed: int = None):
        latent_vector = torch.randn(
            1,
            config.data.latent_vector_size,
            1,
            1,
            device=config.device,
        )
        img = self.model(latent_vector).squeeze().detach().cpu().numpy()
        return np.transpose(denormalize(img), (1, 2, 0))

    def train(
            self,
    ):
        self.logger.info('started dataset new model')
        # prepare data
        dataset = RealImageDataset(
            config.path.training_dataset,
            transform=transforms.Compose(
                [
                    transforms.Resize(config.data.image_size),
                    transforms.CenterCrop(config.data.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            drop_last=True,
        )

        # prepare models
        d_model = models.Discriminator(
            input_size=config.data.image_size,
        ).to(config.device)
        d_model.apply(init_weights)
        g_model = models.Generator(
            input_size=config.data.latent_vector_size,
            output_size=config.data.image_size,
        ).to(config.device)
        g_model.apply(init_weights)

        # link models with optimizers
        d_optimizer = torch.optim.Adam(
            params=d_model.parameters(),
            lr=config.training.d_learning_rate,
            betas=(0.5, 0.9),
        )
        g_optimizer = torch.optim.Adam(
            params=g_model.parameters(),
            lr=config.training.g_learning_rate,
            betas=(0.5, 0.9),
        )

        # prepare to record dataset plots
        d_losses = []
        g_losses = []
        fixed_latent_vector = torch.randn(
            config.training.sample_num,
            config.data.latent_vector_size,
            1,
            1,
            device=config.device,
        )

        # train
        for epoch in range(config.training.epochs):

            print(f'Epoch: {epoch + 1}')
            for idx, (real_images, _) in enumerate(data_loader):

                # show_samples(real_images)

                real_images = real_images.to(config.device)

                print(f'\rProcess: {100 * (idx + 1) / len(data_loader): .2f}%', end='')

                d_loss = None
                for _ in range(config.training.critic_num):
                    d_loss = train_d_model(
                        d_model=d_model,
                        g_model=g_model,
                        real_images=real_images,
                        d_optimizer=d_optimizer,
                    )
                d_losses.append(d_loss)

                g_losses.append(
                    train_g_model(
                        g_model=g_model,
                        d_model=d_model,
                        g_optimizer=g_optimizer,
                    )
                )

            print(
                f"\n"
                f"Discriminator loss: {d_losses[-1]}\n"
                f"Generator loss: {g_losses[-1]}\n"
            )

            # save losses plot
            plt.title("Generator and Discriminator Loss During Training")
            plt.plot(g_losses, label="generator")
            plt.plot(d_losses, label="discriminator")
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(fname=str(config.path.training_plots / 'losses.jpg'))
            plt.clf()

            # save samples
            save_samples(
                file_name=f'E{epoch + 1}.jpg',
                samples=g_model(fixed_latent_vector)
            )

            self.model = g_model
            self.save_model()

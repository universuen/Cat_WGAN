from os import makedirs

from . import config

import torch
from torchvision.utils import save_image, make_grid
from matplotlib import pyplot as plt


def _cal_gradient_penalty(
        d_model: torch.nn.Module,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
):
    alpha = torch.rand(config.training.batch_size, 1, 1, 1).to(config.device)

    interpolates = alpha * real_images + (1 - alpha) * fake_images
    interpolates.requires_grad = True

    disc_interpolates = d_model(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).to(config.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size()[0], -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * config.training.gp_lambda
    return gradient_penalty


def train_g_model(
        g_model: torch.nn.Module,
        d_model: torch.nn.Module,
        g_optimizer: torch.optim.Optimizer,
) -> float:

    # clear the generator's gradients
    g_model.zero_grad()

    # prepare a random latent vector
    l_v = torch.randn(
        config.training.batch_size,
        config.data.latent_vector_size,
        device=config.device,
    )

    # feed the generator with the latent vector to get fake images
    fake_images = g_model(l_v)

    # feed the discriminator with fake images to get its prediction
    prediction = d_model(fake_images)

    # calculate the loss
    # PS: The score should be as high as possible, so the higher the score is,
    # the lower the loss will be.
    loss = - prediction.mean()

    # calculate gradients and update weights
    loss.backward()
    g_optimizer.step()

    return loss.item()


def train_d_model(
        d_model: torch.nn.Module,
        g_model: torch.nn.Module,
        real_images: torch.Tensor,
        d_optimizer: torch.optim.Optimizer,
) -> float:

    # The whole workflow is similar with `train_g_model`

    d_model.zero_grad()

    prediction_real = d_model(real_images)
    loss_real = - prediction_real.mean()

    l_v = torch.randn(
        config.training.batch_size,
        config.data.latent_vector_size,
        device=config.device,
    )
    fake_images = g_model(l_v).detach()
    prediction_fake = d_model(fake_images)
    loss_fake = prediction_fake.mean()

    gradient_penalty = _cal_gradient_penalty(
        d_model=d_model,
        real_images=real_images,
        fake_images=fake_images,
    )

    loss = loss_real + loss_fake + gradient_penalty
    loss.backward()
    d_optimizer.step()

    return loss.item()


def denormalize(image: torch.Tensor) -> torch.Tensor:
    return image * 0.5 + 0.5


def save_samples(
        file_name: str,
        samples: torch.Tensor,
):
    makedirs(config.path.training_plots / 'samples', exist_ok=True)
    save_image(
        denormalize(samples),
        str(config.path.training_plots / 'samples' / file_name),
    )


def show_samples(
        samples: torch.Tensor,
):
    plot = make_grid(
        tensor=denormalize(samples)
    ).permute(1, 2, 0)
    plt.imshow(plot)
    plt.show()
    plt.clf()

import torch
from matplotlib import pyplot as plt
from matplotlib import animation

import context
from generator import Generator
from generator import config

SEEDS = [2, 6, 11, 15, 18]
STEPS = 60
MODEL_NAME = 'cat'
GIF_NAME = 'single_animation_cat.gif'

generator = Generator(MODEL_NAME)
generator.load_model()
images = []
fig = plt.figure()

for i, _ in enumerate(SEEDS):

    torch.manual_seed(SEEDS[i])
    start_vector = torch.randn(1, config.data.latent_vector_size, device=config.device)
    torch.manual_seed(SEEDS[(i + 1) % len(SEEDS)])
    end_vector = torch.randn(1, config.data.latent_vector_size, device=config.device)
    delta = (end_vector - start_vector) / STEPS
    latent_vector = start_vector

    for _ in range(STEPS + 1):
        images.append(
            [
                plt.imshow(
                    generator.generate(
                        latent_vector=latent_vector
                    ),
                    animated=True,
                ),
            ]
        )
        latent_vector += delta

plt.axis("off")
ani = animation.ArtistAnimation(
    fig=fig,
    artists=images,
    interval=30,
    blit=True,
)

ani.save(filename=str(config.path.training_plots / GIF_NAME))
plt.show()
plt.close('all')

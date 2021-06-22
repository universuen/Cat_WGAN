import torch
from matplotlib import pyplot as plt
from matplotlib import animation

import context
from generator import Generator
from generator import config

START_SEED = 555
END_SEED = 777
STEPS = 30
MODEL_NAME = 'cat.model'
GIF_NAME = 'single_animation_cat.gif'

images = []

torch.manual_seed(START_SEED)
start_vector = torch.randn(1, config.data.latent_vector_size, device=config.device)
torch.manual_seed(END_SEED)
end_vector = torch.randn(1, config.data.latent_vector_size, device=config.device)

delta = (end_vector - start_vector) / STEPS
latent_vector = start_vector

fig = plt.figure()

generator = Generator(MODEL_NAME)
generator.load_model()

for j in range(STEPS + 1):
    images.append(
        [
            plt.imshow(
                generator.generate(
                    latent_vector=latent_vector
                ),
                animated=True
            )
        ]
    )
    latent_vector += delta

plt.axis("off")
ani = animation.ArtistAnimation(
    fig=fig,
    artists=images + images[::-1],
    interval=30,
    blit=True,
)

ani.save(filename=str(config.path.training_plots / GIF_NAME))
plt.show()
plt.close('all')

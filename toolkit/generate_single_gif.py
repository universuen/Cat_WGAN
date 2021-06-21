import torch
from matplotlib import pyplot as plt
from matplotlib import animation

import context
from pokemon_generator import PokemonGenerator
from pokemon_generator import config

START_SEED = 666
END_SEED = 777
STEPS_NUM = 100
MODEL_NAME = 'pokemon.model'
GIF_NAME = 'single_animation_pokemon.gif'

images = []

torch.manual_seed(START_SEED)
start_vector = torch.randn(1, config.data.latent_vector_size, 1, 1, device=config.device)
torch.manual_seed(END_SEED)
end_vector = torch.randn(1, config.data.latent_vector_size, 1, 1, device=config.device)

delta = (end_vector - start_vector) / STEPS_NUM
latent_vector = start_vector

fig = plt.figure()

generator = PokemonGenerator(MODEL_NAME)
generator.load_model()

for j in range(STEPS_NUM + 1):
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
    artists=images,
    interval=100,
    blit=True,
)

ani.save(filename=str(config.path.training_plots / GIF_NAME))
plt.show()
plt.close('all')

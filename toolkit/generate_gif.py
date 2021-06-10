from os import listdir

from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.image import imread

import context
from pokemon_generator.config.path import training_plots as plots_path

images_dir = plots_path / 'samples'
images = []

fig = plt.figure()

cnt = -1
for file_name in sorted(listdir(images_dir), key=lambda x: int(x[1:-4])):
    cnt += 1
    if cnt % 10 != 0:
        continue
    images.append(
        [
            plt.imshow(
                imread(str(images_dir / file_name)),
                animated=True,
            )
        ]
    )

plt.axis("off")
ani = animation.ArtistAnimation(
    fig=fig,
    artists=images,
    interval=100,
    blit=True,
)

ani.save(filename='animation.gif')
plt.show()
plt.close('all')

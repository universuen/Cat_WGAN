from os import listdir

from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.image import imread

import context
from generator.config import path
from generator.config.training import batch_size


GIF_NAME = 'training_animation_cat.gif'
STEP = 1

images_dir = path.training_plots / 'samples'
images = []

dataset_size = len(listdir(path.training_dataset))

fig = plt.figure()

cnt = -1

for file_name in sorted(listdir(images_dir), key=lambda x: int(x[1:-4])):

    cnt += 1

    if cnt % STEP != 0:
        continue

    epoch = file_name[1:][:-4]

    images.append(
        [
            plt.imshow(
                imread(str(images_dir / file_name)),
                animated=True,
            ),
            plt.text(0, -10, f'epoch: {epoch}, iteration: {int(epoch) * (dataset_size // batch_size)}')
        ]
    )

plt.axis("off")
ani = animation.ArtistAnimation(
    fig=fig,
    artists=images,
    interval=100,
    blit=False,
)

ani.save(filename=str(path.training_plots / GIF_NAME))
plt.show()
plt.close('all')

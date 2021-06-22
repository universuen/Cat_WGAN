# Pokemon WGAN

Generate your unique Pokemon with one click!

# Preview
![](https://github.com/universuen/Cat_WGAN/blob/main/data/training/plots/losses.jpg)
![](https://github.com/universuen/Cat_WGAN/blob/main/data/training/plots/training_animation_cat.gif)
![](https://github.com/universuen/Cat_WGAN/blob/main/data/training/plots/single_animation_cat.gif)

# Usage
1. `pip install -r requirements.txt`
2. `py run.py`

# Additional Info
This project is portable for any legal image dataset, which means you
can train your own model on a given dataset.

To train a new model, extract images into `dara/training/dataset`, then
configure `pokemon_generator.config.training` and `pokemon_generator.config.data`
according to your own preferences. Additionally, don't forget to write your model
name in `train.py` and `run.py`.

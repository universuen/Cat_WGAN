from matplotlib import pyplot as plt

from pokemon_generator import PokemonGenerator, config

if __name__ == '__main__':
    config.device = 'cpu'
    generator = PokemonGenerator(
        model_name='cat.model',
    )
    generator.load_model()
    img = generator.generate()
    plt.axis('off')
    plt.imshow(img)
    plt.show()

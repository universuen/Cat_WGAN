from matplotlib import pyplot as plt

from pokemon_generator import PokemonGenerator


if __name__ == '__main__':
    generator = PokemonGenerator(
        model_name='cat.model',
    )
    generator.load_model()
    img = generator.generate()
    plt.axis('off')
    plt.imshow(img)
    plt.show()

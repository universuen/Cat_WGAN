from matplotlib import pyplot as plt

from generator import Generator


if __name__ == '__main__':
    generator = Generator(
        model_name='cat.model',
    )
    generator.load_model()
    img = generator.generate(777)
    plt.axis('off')
    plt.imshow(img)
    plt.show()

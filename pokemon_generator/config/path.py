from pathlib import Path

src = Path(__file__).absolute().parent.parent
project = src.parent
data = project / 'data'
models = data / 'models'
training_data = data / 'training'

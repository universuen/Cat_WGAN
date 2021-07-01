from pathlib import Path

src = Path(__file__).absolute().parent.parent
project = src.parent
data = project / 'data'
models = data / 'models'
training_dataset = data / 'training' / 'dataset'
training_plots = data / 'training' / 'plots'
checkpoint = data / 'training' / 'checkpoint.bin'

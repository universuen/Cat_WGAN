import context
from pokemon_generator.logger import Logger


def test_debug():
    Logger('test').debug('Hello')


def test_info():
    Logger('test').info('Hello')


def test_warning():
    Logger('test').warning('Hello')


def test_error():
    Logger('test').error('Hello')


def test_fatal():
    Logger('test').fatal('Hello')

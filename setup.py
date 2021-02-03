from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='tweet-se-competition',
    packages=find_packages(),
    version='0.5.0',
    description='a machine learning project for kaggle tweet sentiment extraction competition',
    author='Simon Meoni',
    license='MIT',
    install_requires=requirements,
    entry_points='''
    [console_scripts]
    distillbert_tokens_classification_train=src.models.distillbert_tokens_classification_train:main
    ''',
)

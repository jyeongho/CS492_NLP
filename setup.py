# nsml: nsml/ml:cuda10.1-cudnn7-pytorch1.3keras2.3

from distutils.core import setup

setup(
    name='kaist-korquad-test',
    version='1.0',
    install_requires=[
        'boto3', 'regex', 'sacremoses', 'filelock', 'tokenizers',
        'tqdm', 'konlpy', 'sentencepiece', 'dataclasses',
        'transformers==3.5.1', 'torch==1.7.0', 'torchvision==0.8.1',
    ]
)

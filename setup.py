import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


setup(
    name="se_eval",
    version="0.1a",
    url="https://github.com/MandarGogate/se_eval",
    license='MIT',

    author="Mandar Gogate",
    author_email="m.gogate@napier.ac.uk",

    description="A fast wrapper around several freely available implementations of objective metrics for speech enhancement and separation",
    long_description=read("README.rst"),

    packages=find_packages(exclude=('tests',)),

    install_requires=[
        "torchmetrics[audio]>=0.8.0",
        "torchaudio",
        "librosa",
        "tqdm"
    ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)

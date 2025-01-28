# -*- coding: utf-8 -*-

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="pywph",
    version="1.1.3",
    author="Bruno Régaldo-Saint Blancard",
    author_email="bregaldosaintblancard@flatironinstitute.org",
    description="Wavelet Phase Harmonics in Python with GPU acceleration.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bregaldo/pywph",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)

# setup.py
from setuptools import setup, find_packages

setup(
    name="probstates",
    version="0.2.0",
    author="Ed Chrednik",
    author_email="edchrednik@gmail.com",
    description="Библиотека для работы с формализмом вероятностных состояний",
    # Read long description from readme (case-insensitive on most dev machines,
    # but use lowercase filename that exists in repo)
    long_description=open("readme.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/probstates/probstates",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.18.0",
        "matplotlib>=3.3.0",
    ],
)

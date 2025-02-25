# setup.py
from setuptools import setup, find_packages

setup(
    name="probstates",
    version="0.1.0",
    author="ProbStates Team",
    author_email="info@probstates.org",
    description="Библиотека для работы с формализмом вероятностных состояний",
    long_description=open("README.md").read(),
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

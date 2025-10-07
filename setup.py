from setuptools import setup, find_packages
from pathlib import Path

reqs = Path("requirements.txt").read_text().splitlines()

setup(
    name="brainchop",
    version="0.1.23",
    author="Mike Doan",
    author_email="spikedoanz@gmail.com",
    description="Portable and lightweight brain segmentation using tinygrad",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/neuroneural/brainchop-cli",
    packages=find_packages(),
    setup_requires=[
        "pybind11>=2.5.0",
    ],
    install_requires=reqs,
    entry_points={
        "console_scripts": [
            "brainchop=brainchop.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "brainchop": ["niimath/*"],
        "multiaxial_brain_segmenter": ["models/*.onnx"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="brainchop",
    version="0.1.4",
    author="Mike Doan",
    author_email="spikedoanz@gmail.com",
    description="Portable and lightweight brain segmentation using tinygrad",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neuroneural/brainchop-cli",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tinygrad",
        "nibabel",
        "requests",
    ],
    entry_points={
        "console_scripts": [
            "brainchop=brainchop.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "brainchop": ["model.json", "model.bin"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

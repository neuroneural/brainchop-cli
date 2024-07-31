from setuptools import setup, find_packages

setup(
    name="brainchop",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tinygrad",
        "nibabel",
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
)

from setuptools import setup, find_packages, Extension
from setuptools.dist import Distribution

setup_requires = ['numpy']

# Attempt to import numpy, but don't fail if it's not available
try:
    import numpy
    numpy_include = [numpy.get_include()]
except ImportError:
    numpy_include = ['.', 'brainchop/utils/nd_image']

# Define the extension module
# btw maybe run python setup.py build_ext --inplace
nd_image_module = Extension('brainchop.utils._nd_image',
                            sources=['brainchop/utils/nd_image/nd_image.c',
                                     'brainchop/utils/nd_image/ni_filters.c',
                                     'brainchop/utils/nd_image/ni_fourier.c',
                                     'brainchop/utils/nd_image/ni_interpolation.c',
                                     'brainchop/utils/nd_image/ni_measure.c',
                                     'brainchop/utils/nd_image/ni_morphology.c',
                                     'brainchop/utils/nd_image/ni_splines.c',
                                     'brainchop/utils/nd_image/ni_support.c'],
                            include_dirs=numpy_include,
                            extra_compile_args=['-std=c99'])

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="brainchop",
    version="0.1.5",
    author="Mike Doan",
    author_email="spikedoanz@gmail.com",
    description="Portable and lightweight brain segmentation using tinygrad",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neuroneural/brainchop-cli",
    packages=find_packages(),
    ext_modules=[nd_image_module],
    setup_requires=setup_requires,
    install_requires=[
        'numpy',
        'tinygrad',
        'requests',
        'nibabel',
    ],
    entry_points={
        "console_scripts": [
            "brainchop=brainchop.main:main",
        ],
    },
    include_package_data=True,
    package_data={},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

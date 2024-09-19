from setuptools import setup, find_packages
import os
import platform
import shutil

def get_niimath_path():
    system = platform.system().lower()
    if system == 'linux':
        return os.path.join('brainchop', 'niimath', 'linux', 'niimath')
    elif system == 'darwin':
        return os.path.join('brainchop', 'niimath', 'macos', 'niimath')
    elif system == 'windows':
        return os.path.join('brainchop', 'niimath', 'windows', 'niimath.exe')
    else:
        raise OSError(f"Unsupported operating system: {system}")

def copy_niimath_executable(setup_kwargs):
    niimath_src = get_niimath_path()
    niimath_dest = os.path.join('brainchop', 'niimath')
    os.makedirs(niimath_dest, exist_ok=True)
    shutil.copy2(niimath_src, niimath_dest)

setup(
    name="brainchop",
    version="0.1.6",
    author="Mike Doan",
    author_email="spikedoanz@gmail.com",
    description="Portable and lightweight brain segmentation using tinygrad",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/neuroneural/brainchop-cli",
    packages=find_packages(),
    setup_requires=[
        'pybind11>=2.5.0',
    ],
    install_requires=[
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
    package_data={
        'brainchop': ['niimath/*'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

if __name__ == '__main__':
    copy_niimath_executable(locals())

import os
from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
try:
  README = open(os.path.join(here, "README.md")).read()
except IOError:
  README = ""

version = "0.0.1"

install_requires = [
  "numpy>=1.12",
  "jaxlib>=0.1.41",
  "jax>=0.1.59",
  "flax",
  "matplotlib",
  "dataclasses",
  "msgpack",
]

tests_require = [
]

setup(
    name="ladax",
    version=version,
    description="Ladax: layered distribution models using FLAX/JAX.",
    long_description="\n\n".join([README]),
    long_description_content_type='text/markdown',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    keywords="",
    author="Dan Tait",
    author_email="tait.djk@gmail.com",
    url="https://github.com/danieljtait/ladax",
    license="Apache",
    packages=find_packages(),
    include_package_data=False,
    zip_safe=False,
    install_requires=install_requires,
    extras_require={
        "testing": tests_require,
        },
    )

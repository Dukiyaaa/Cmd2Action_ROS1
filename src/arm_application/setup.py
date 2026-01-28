from setuptools import setup, find_packages
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=find_packages(where='arm_application'),
    package_dir={'': 'arm_application'}
)
setup(**d)
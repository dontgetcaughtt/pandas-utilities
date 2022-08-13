from setuptools import setup

from pdutils import __version__

setup(
    name='pandas-utils',
    version=__version__,
    packages=['pdutils'],
    url='https://github.com/dontgetcaughtt/pd-utils',
    license='MIT',
    author='Kristoffer Thom',
    author_email='dontgetcaughtt@outlook.com',
    description='More or less useful utilities for pandas',
    python_requires='>=3.10',
    install_requires='pandas'
)

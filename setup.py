from setuptools import setup

setup(
    name='imantics',
    version='0.0.0',
    description='Python package for managing image annotations',
    url='https://github.com/jsbroks/imantics',
    author='Justin Brooks',
    author_email='jsbroks@gmail.com',
    license='MIT',
    install_requires=['numpy', 'opencv-python>=3'],
    packages=['imantics'],
    zip_safe=False
)

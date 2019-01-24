from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='imantics',
    version='0.1.0',
    description='Python package for managing image annotations',
    url='https://github.com/jsbroks/imantics',
    author='Justin Brooks',
    author_email='jsbroks@gmail.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    install_requires=['numpy', 'opencv-python>=3', 'lxml'],
    packages=setup.find_packages(),
    zip_safe=False
)

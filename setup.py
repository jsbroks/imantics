from setuptools import setup
import subprocess


def get_tag():
    result = subprocess.run(["git", "describe", "--abbrev=0", "--tags"], stdout=subprocess.PIPE)
    return str(result.stdout.decode("utf-8")).strip()[1:]


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='imantics',
    version=get_tag(),
    description='Python package for managing image annotations',
    url='https://github.com/jsbroks/imantics',
    author='Justin Brooks',
    author_email='jsbroks@gmail.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    install_requires=['numpy', 'opencv-python>=3', 'lxml'],
    packages=['imantics'],
    python_requires='>=3',
    zip_safe=False
)

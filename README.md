# Image Semantics

<p align="center">
  <a href="/jsbroks/imantics/stargazers">
    <img src="https://img.shields.io/github/stars/jsbroks/imantics.svg">
  </a>
  <a href="/jsbroks/imantics/issues">
    <img src="https://img.shields.io/github/issues/jsbroks/imantics.svg">
  </a>
  <a href="https://tldrlegal.com/license/mit-license">
    <img src="https://img.shields.io/github/license/mashape/apistatus.svg">
  </a>
  <a href="https://travis-ci.org/jsbroks/imantics">
    <img src="https://travis-ci.org/jsbroks/imantics.svg?branch=master">
  </a>
  <a href="https://imantics.readthedocs.io/en/latest/?badge=latest">
    <img src="https://readthedocs.org/projects/imantics/badge/?version=latest">
  </a>
  <a href="https://pypi.org/project/imantics/">
    <img src="https://img.shields.io/pypi/v/imantics.svg">
  </a>
  <a href="https://pypi.org/project/imantics/">
    <img src="https://img.shields.io/pypi/dm/imantics.svg">
  </a>
</p>

Image understanding is widely used in many areas like satellite imaging, robotic technologies, sensory networks, medical and biomedical imaging, intelligent transportation systems, etc. Recently semantic analysis has become an active research topic aimed at resolving the gap between low level image features and high level semantics which is a promoting approach in image understanding.

With many image annotation semantics existing in the field of computer vision, it can become daunting to manage. This package provides the ability to convert and visualize many different types of annotation formats for object dectection and localization.

Currently Support Formats:

- COCO Format
- Binary Masks
- YOLO
- VOC

This version fix color order changes from BGR to RGB introduced in version 0.1.12 (or at least after 0.1.9).  
This version also fix infinite loop in Annotation.coco() thank's to george-gca work. (cherry-pick george-gca fix).
Also note that v0.1.12 is actually the official imantics pip version, but this version is buggous.  
Stay on master branch to benefit from unreleased 0.1.13 version that fix problems (except color order change and infinite loop)

so instead of 
```
pip install imantics
```
prefer
```
pip install imantics==0.1.9

or better, get version with my fixes
pip install git+https://github.com/SixK/imantics.git
```


<p align="center">Join our growing <a href="https://discord.gg/4zP5Qkj">discord community</a> of ML practitioner</p>
<p align="center">
  <a href="https://discord.gg/4zP5Qkj">
    <img src="https://discord.com/assets/e4923594e694a21542a489471ecffa50.svg" width="120">
  </a>
</p>

<br />

<p align="center">If you enjoy my work please consider supporting me</p>
<p align="center">
  <a href="https://www.patreon.com/jsbroks">
    <img src="https://c5.patreon.com/external/logo/become_a_patron_button@2x.png" width="120">
  </a>
</p>

## Installing

```
pip install imantics
```

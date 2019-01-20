import numpy as np

from .styles import *
from .image import Image
from .basic import Semantic


class Dataset(Semantic):
    
    @classmethod
    def from_coco(cls):
        pass
    
    def __init__(self, name, images, id=0, metadata={}):
        self.name = name

        self.annotations = {}
        self.categories = {}
        self.images = {}

        self._index_image(images)
        super().__init__(id, metadata)
    
    def add(self, image):

        if isinstance(image, str):
            image = Image.from_path(image)
        
        self._index_image(image)
    
    def _index_image(self, image):

        if isinstance(image, (list, tuple)):
            for img in image:
                self._index_image(img)
            return

        print(image)
    
    def _coco(self, include=True):
        pass
    
    def _yolo(self):
        pass


__all__ = ["Dataset"]
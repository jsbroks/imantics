import numpy as np

from .image import Image
from .basic import Semantic


class Dataset(Semantic):
    
    @classmethod
    def from_coco(cls):
        pass
    
    annotations = {}
    categories = {}
    images = {}
    
    def __init__(self, name, images, id=0, metadata={}):
        self.name = name

        for image in images:
            image.index(self)
        
        super().__init__(id, metadata)
    
    def add(self, image):
        """
        Adds image(s) to the current dataset

        :param image: image, list of images, or path to image(s)
        """
        if isinstance(image, str):
            image = Image.from_path(image)
        
        if isinstance(image, (list, tuple)):
            for img in image:
                img.index(self)
        
        image.index(self)

    def _coco(self, include=True):
        pass
    
    def _yolo(self):
        pass


__all__ = ["Dataset"]
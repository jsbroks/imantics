import cv2
import os
import numpy as np
from .annotation import BBox, Mask, Polygons, Annotation


class Image:

    @classmethod
    def from_coco(cls, coco):
        pass

    @classmethod
    def from_path(cls, path):
        brg = cv2.imread(path)
        image_array = cv2.cvtColor(brg, cv2.COLOR_BGR2RGB)
        return cls(image_array, path=path)

    @classmethod
    def empty(cls):
        return cls([[[]]])

    def __init__(self, image_array, annotations=[], path=""):
        
        self.annotations = {}
        # index annotation
        for index, annotation in enumerate(annotations):
            annotation.id = index+1
            self._index_annotation(annotation)

        self.array = np.array(image_array)
        self.path = path

        self.height, self.width, _ = self.array.shape
        self.size = (self.width, self.height)
        self.file_name = os.path.basename(self.path)

    def add(self, annotation, category=None):
        if isinstance(annotation, Mask):
            annotation = Annotation.from_mask(self, category, annotation)

        if isinstance(annotation, BBox):
            annotation = Annotation.from_bbox(self, category, annotation)

        if isinstance(annotation, Polygons):
            annotation = Annotation.from_polygons(self, category, annotation)
            
        self._index_annotation(annotation)

    def _index_annotation(self, annotation):
        
        if annotation.id < 1:
            annotation.id = len(self.annotations) + 1
        
        found = self.annotations.get(annotation.id, None)
        if found:
            annotation.id = annotation.id + 1
            self._index_annotation(annotation)
        else:
            self.annotations[annotation.id] = annotation

    def draw_masks(self, alpha=0.5):

        temp_image = self.array.copy()
        temp_image.setflags(write=True)

        for key, annotation in self.annotations.items():
            if isinstance(key, int):
                annotation.mask.apply(temp_image, alpha=alpha)
        
        return temp_image
        
    def draw_bboxs(self, thickness=3):

        temp_image = self.array.copy()
        temp_image.setflags(write=True)

        for key, annotation in self.annotations.items():
            if isinstance(key, int):
                annotation.bbox.apply(temp_image, thickness=thickness)
            
        return temp_image

    def export(self):
        pass



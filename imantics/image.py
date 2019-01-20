
import os
import cv2
import json
import numpy as np

from .annotation import *
from .utils import json_default
from .styles import COCO, VGG, VOC, YOLO


class Image:
    
    FORMATS = ('.png', '.jpg', '.jpeg', '.jpe', '.tiff', '.bmp', '.sr', '.ras')

    @classmethod
    def from_folder(cls, directory):
        images = []
        for path, _, files in os.walk(directory):
            for name in files:
                file_path = os.path.join(path, name)
                if file_path.lower().endswith(cls.FORMATS):
                    images.append(cls.from_path(file_path))
        
        return images

    @classmethod
    def from_path(cls, path):
        brg = cv2.imread(path)
        image_array = cv2.cvtColor(brg, cv2.COLOR_BGR2RGB)
        return cls(image_array, path=path)

    @classmethod
    def empty(cls):
        return cls([[[]]])

    def __init__(self, image_array, annotations=[], path="", id=0, metadata={}):
        self.id = id

        self.metadata = metadata
        self.annotations = {}
        self.categories = {}

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
        
        found = self.annotations.get(annotation.id)
        if found:
            annotation.id = annotation.id + 1
            self._index_annotation(annotation)
        else:
            category = annotation.category
            self.annotations[annotation.id] = annotation
            self.categories[category.name] = category

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

    def _coco(self, include=True):
        image = {
            'id': self.id,
            'width': self.width,
            'height': self.height,
            'file_name': self.file_name,
            'path': self.path,
            'license': self.metadata.get('license'),
            'fickr_url': self.metadata.get('flicker_url'),
            'coco_url': self.metadata.get('coco_url'),
            'date_captured': self.metadata.get('date_captured'),
            'metadata': self.metadata
        }
        
        if include:
            
            categories = []
            for _, category in self.categories.items():
                category.id = len(categories) + 1
                categories.append(category._coco(include=False))

            annotations = []
            for key, annotation in self.annotations.items():
                if isinstance(key, int):
                    annotation.id = len(annotations) + 1
                    annotations.append(annotation._coco(include=False))

            return {
                'categories': categories,
                'images': [image],
                'annotations': annotations
            }

        return coco
    
    def _yolo(self):
        yolo = []
        
        categories = []
        for _, category in self.categories.items():
            category.id = len(categories) + 1
            categories.append(category)
        
        for key, annotation in self.annotations.items():
            if isinstance(key, int):
                yolo.append(annotation._yolo())
        
        return yolo

    def export(self, style=COCO):
        return {
            COCO: self._coco(),
            VGG: None,
            VOC: None,
            YOLO: self._yolo()
        }.get(style)

    def save(self, file_path, style=COCO):
        with open(file_path, 'w') as fp:
            json.dump(self.export(style=style), fp)


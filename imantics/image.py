
import os
import cv2
import json
import numpy as np

from .annotation import *
from .basic import Semantic
from .utils import json_default
from .styles import COCO, VGG, VOC, YOLO


class Image(Semantic):
    
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
        """
        Returns an array of images if path is a directory
        Returns an image if path is a file
        """
        if os.path.isdir(path):
            return Image.from_folder(path)
        
        brg = cv2.imread(path)
        image_array = cv2.cvtColor(brg, cv2.COLOR_BGR2RGB)

        return cls(image_array, path=path)

    @classmethod
    def from_coco(cls, coco, dataset=None):

        metadata = coco.get('metadata', {})

        metadata.update({
            'license': coco.get('license'),
            'flickr_url': coco.get('flickr_url'),
            'coco_url': coco.get('coco_url'),
            'date_captured': coco.get('date_captured')
        })

        data = {
            'id': coco.get('id', 0),
            'width': coco.get('width', 0),
            'height': coco.get('height', 0),
            'path': coco.get('path', coco.get('file_name', '')),
            'metadata': metadata,
            'dataset': dataset
        }

        return cls(**data)

    @classmethod
    def empty(cls, width=0, height=0):
        return cls(width=width, height=height)

    annotations = {}
    categories = {}

    def __init__(self, image_array=None, annotations=[], path="", id=0, metadata={}, dataset=None, width=0, height=0):

        self.dataset = dataset
        self.annotations = {}
        self.categories = {}

        # Index annotation
        for index, annotation in enumerate(annotations):
            annotation.id = index+1
            annotation.index(self)

        self.path = path
        if image_array is None:
            self.array = np.zeros((height, width, 3)).astype(np.uint8)
        else:
            self.array = image_array
        
        self.height, self.width, _ = self.array.shape
        
        self.size = (self.width, self.height)
        self.file_name = os.path.basename(self.path)

        super().__init__(id, metadata)

    def add(self, annotation, category=None):
        """
        Adds a annotaiton, list of annotaitons, mask, polygon or bbox to current image.
        If annotation is not a Annotation a category is required
        List of non-Annotaiton objects will have the same category

        :param annotation: annotaiton to add to current image
        :param category: required if annotation is not an Annotation object
        """
        if isinstance(annotation, list):
            for ann in annotation:
                self.add(ann)
            return

        if isinstance(annotation, Mask):

            height, width = annotation.array.shape[:2]
            if width == self.width and height == self.height:
                annotation = Annotation.from_mask(self, category, annotation)
            else:
                raise ValueError('Cannot add annotaiton of size {} to image of size {}'\
                                 .format(annotation.array.shape, (self.height, self.width)))
            
        if isinstance(annotation, BBox):
            annotation = Annotation.from_bbox(self, category, annotation)

        if isinstance(annotation, Polygons):
            annotation = Annotation.from_polygons(self, category, annotation)
        
        annotation.index(self)

    def index(self, dataset):
        
        image_index = dataset.images
        image_index[self.id] = self

        for annotation in self.annotations:
            annotation.index(dataset)

    def draw_masks(self, image=None, alpha=0.5, categories=None, color_by_category=False):

        temp_image = image.copy() if image is not None else self.array.copy()
        temp_image.setflags(write=True)
        
        for _, annotation in self.annotations.items():
            category = annotation.category
            if  (categories is None) or (category in categories):
                color = category.color if color_by_category else annotation.color
                annotation.mask.draw(temp_image, alpha=alpha, color=color)
        
        return temp_image
        
    def draw_bboxs(self, image=None, thickness=3, categories=None, color_by_category=False):

        temp_image = image.copy() if image is not None else self.array.copy()
        temp_image.setflags(write=True)

        for _, annotation in self.annotations.items():
            category = annotation.category
            if  (categories is None) or (category in categories):
                color = category.color if color_by_category else annotation.color
                annotation.bbox.draw(temp_image, thickness=thickness, color=color)
        
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
            for _, annotation in self.annotations.items():
                annotation.id = len(annotations) + 1
                annotations.append(annotation._coco(include=False))

            return {
                'categories': categories,
                'images': [image],
                'annotations': annotations
            }

        return image
    
    def _yolo(self):
        yolo = []
        
        categories = []
        for _, category in self.categories.items():
            category.id = len(categories) + 1
            categories.append(category)
        
        for key, annotation in self.annotations.items():
            yolo.append(annotation._yolo())
        
        return yolo

    def save(self, file_path, style=COCO):
        with open(file_path, 'w') as fp:
            json.dump(self.export(style=style), fp)


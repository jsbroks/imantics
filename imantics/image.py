
from lxml import etree as ET    
from lxml.builder import E

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
        """
        Creates :class:`Image`'s from all images found in directory

        :returns: list of :class:`Image`'s
        """
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
        Returns an :class:`Image` if path is a file
        """
        if os.path.isdir(path):
            return Image.from_folder(path)
        
        brg = cv2.imread(path)
        image_array = cv2.cvtColor(brg, cv2.COLOR_BGR2RGB)

        return cls(image_array, path=path)

    @classmethod
    def from_coco(cls, coco, dataset=None):
        """
        Creates an :class:`Image` from a dict in COCO formatted image

        :param coco: COCO formatted image
        :type coco: dict
        :rtype: :class:`Image`
        """
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
        """
        Creates an empty :class:`Image`
        """
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
        # Load image form path if path is provided and image_array is not
        #if len(path) != 0 and image_array is None:
            # self = Image.from_path(path)
        
        # Create empty image if not provided
        if image_array is None:
            self.height, self.width = (height, width)
        else:
            self.height, self.width, _ = image_array.shape
        
        self.size = (self.width, self.height)
        self.file_name = os.path.basename(self.path)

        super(Image, self).__init__(id, metadata)

    def add(self, annotation, category=None):
        """
        Adds an annotation, list of annotation, mask, polygon or bbox to current image.
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
                annotation = Annotation.from_mask(annotation, image=self, category=category)
            else:
                raise ValueError('Cannot add annotaiton of size {} to image of size {}'\
                                 .format(annotation.array.shape, (self.height, self.width)))
            
        if isinstance(annotation, BBox):
            annotation = Annotation.from_bbox(annotation, image=self, category=category)

        if isinstance(annotation, Polygons):
            annotation = Annotation.from_polygons(annotation, image=self, category=category)
        
        annotation.set_image(self)
        annotation.index(self)

    def index(self, dataset):
        
        image_index = dataset.images
        image_index[self.id] = self

        for annotation in self.iter_annotations():
            annotation.index(dataset)

    def draw(self, bbox=True, outline=True, mask=True, text=True, thickness=3, \
             alpha=0.5, categories=None, text_scale = 0.5, color_by_category=False):
        """
        Draws annotations on top of the image. If no image is loaded, annotations will be applied
        to a black image array.
        
        :param bbox: Draw bboxes
        :param outline: Draw mask outlines
        :param mask: Draw masks
        :param alpha: opacity of masks (only applies to masks)
        :param thickness: pixel width of lines for outline and bbox
        :param color_by_category: Use the annotations's category to us as color
        :param categories: List of categories to show
        :returns: Image array with annotations
        :rtype: numpy.ndarray
        """
        
        
        temp_image = cv2.imread(self.path)
        if temp_image is None:
            temp_image = np.zeros((self.height,self.width,3)).astype(np.uint8)
        temp_image.setflags(write=True)

        for annotation in self.iter_annotations():
            category = annotation.category
            if  (categories is None) or (category in categories):
                color = category.color if color_by_category else annotation.color

                if mask:
                    annotation.mask.draw(temp_image, alpha=alpha, color=color)
                
                if outline:
                    annotation.polygons.draw(temp_image, color=color, thickness=thickness)

                if bbox:
                    annotation.bbox.draw(temp_image, thickness=thickness, color=color)
                if text:
                    cv2.putText(temp_image, category.name, annotation.bbox.top_left, 
                        cv2.FONT_HERSHEY_PLAIN, text_scale, (0,0,0), 2, cv2.LINE_AA)
                    cv2.putText(temp_image, category.name, annotation.bbox.top_left, 
                        cv2.FONT_HERSHEY_PLAIN, text_scale, (255,255,255), 1, cv2.LINE_AA)

        return temp_image       

    def iter_annotations(self):
        """
        Generator to iterate over all annotations
        """
        for key, annotation in self.annotations.items():
            if isinstance(key, int):
                yield annotation

    def iter_categories(self):
        """
        Generator to iterate over all categories
        """
        for key, category in self.categories.items():
            yield category
        
    def coco(self, include=True):
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
            for category in self.iter_categories():
                category.id = len(categories) + 1
                categories.append(category.coco(include=False))

            annotations = []
            for annotation in self.iter_annotations():
                annotation.id = len(annotations) + 1
                annotations.append(annotation.coco(include=False))

            return {
                'categories': categories,
                'images': [image],
                'annotations': annotations
            }

        return image
    
    def yolo(self):
        yolo = []
        
        categories = []
        for category in self.iter_categories():
            category.id = len(categories) + 1
            categories.append(category)
        
        for annotation in self.iter_annotations():
            yolo.append(annotation.yolo())
        
        return yolo

    def voc(self, pretty=False):

        annotations = []
        for annotation in self.iter_annotations():
            annotations.append(annotation.voc())
        
        element = E('annotation',
            E('folder', self.path[: -1*(len(self.file_name)+1)]),
            E('path', self.path),
            E('filename', self.file_name),
            E('size',
                E('width', str(self.width)),
                E('height', str(self.height)),
                E('depth', str(3))
            ),
            *annotations
        )
        
        if pretty:
            return ET.tostring(element, pretty_print=True).decode('utf-8')
        
        return element

    def save(self, file_path, style=COCO):
        with open(file_path, 'w') as fp:
            json.dump(self.export(style=style), fp)


import numpy as np

from .annotation import Annotation
from .category import Category
from .basic import Semantic
from .image import Image


class Dataset(Semantic):
    
    @classmethod
    def from_coco(cls, coco_obj, name="COCO Datset"):
        """
        Generates a dataset from a COCO object or python dict

        :param coco_obj: 
        :type coco_obj: dict, pycocotools.coco.COCO
        :raise ImportError: Raised if coco_obj is a `pycocotools.coco.COCO`
                            object and it cannot be imported
        """
        if isinstance(coco_obj, dict):
            dataset = cls(name)
            
            coco_info = coco_obj.get('info', [])
            coco_annotations = coco_obj.get('annotations', [])
            coco_images = coco_obj.get('images', [])
            coco_categories = coco_obj.get('categories', [])

            index_categories = {}
            for category in coco_categories:
                category = Category.from_coco(category)
                index_categories[category.id] = category

            for image in coco_images:
                image = Image.from_coco(image)
                dataset.add(image)

            for annotation in coco_annotations:
                
                image_id = annotation.get('image_id')
                category_id = annotation.get('category_id')

                image = dataset.images[image_id]
                category = index_categories[category_id]
                segmentation = annotation.get('segmentation')

                annotation = Annotation(image, category, polygons=segmentation)
                dataset.add(annotation)
            
            return dataset
        
        from pycocotools.coco import COCO
        if isinstance(coco_obj, COCO):
            pass
    
    annotations = {}
    categories = {}
    images = {}
    
    def __init__(self, name, images=[], id=0, metadata={}):
        self.name = name

        for image in images:
            image.index(self)
        
        super().__init__(id, metadata)
    
    def add(self, image):
        """
        Adds image(s) to the current dataset

        :param image: list, object or path to add to dataset
        :type image: :class:`Image` :class:`Annotation`, list, typle, path
        """
        if isinstance(image, (list, tuple)):
            for img in image:
                img.index(self)
            return
        
        if isinstance(image, Annotation):
            annotation = image
            image = self.images.get(annotation.image.id)

            annotation.index(self)
            image.add(annotation)
            return

        if isinstance(image, str):
            image = Image.from_path(image)
                
        image.index(self)
    
    def split(self, ratios, random=False):
        """
        Splits dataset images into mutiple sub datasets of the given ratios

        If a tuple of (1, 1, 2) was passed in the result would return 3 dataset
        objects of 25%, 25% and 50% of the images.

        .. code-block:: python

            percents = ratios / ratios.sum()

        :param ratios: ratios to split dataset into
        :type ratios: tuple, list
        :param random: randomize the images before spliting
        :returns: tuple of datasets with length of the number of ratios
        :rtype: tuple
        """
        ratios = np.array(ratios)
        percents = ratios / ratios.sum()

        if percents.sum() == 100:
            percents /= 100

        print(percents)

    def _coco(self, include=True):
        pass
    
    def _yolo(self):
        pass


__all__ = ["Dataset"]

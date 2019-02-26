from .styles import *


class Semantic(object):

    def __init__(self, id, metadata={}):
        self.id = id
        self.metadata = metadata
    
    def coco(self):
        """
        Export object in COCO format
        
        :returns: object in format
        :rtype: dict
        """
        return {}
    
    def vgg(self):
        """
        Export object in VGG format
        """
        return []
    
    def voc(self):
        """
        Export object in VOC format

        :returns: object in format
        :rtype: lxml.element
        """
        return None
    
    def yolo(self):
        """
        Export object in YOLO format

        :returns: object in format
        :rtype: list, tuple
        """
        return []
    
    def paperjs(self):
        """
        Export object in PaperJS format

        :returns: object in format
        :rtype: dict
        """
        return {}
    
    def export(self, style=COCO):
        """
        Exports object into specified style
        """
        return {
            COCO: self.coco(),
            VGG: self.vgg(),
            YOLO: self.yolo(),
            VOC: self.voc(),
            PAPERJS: self.paperjs()
        }.get(style)
    
    def save(self, file):
        pass

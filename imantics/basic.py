from .styles import *


class Semantic:

    def __init__(self, id, metadata={}):
        self.id = id
        self.metadata = metadata
    
    def coco(self):
        return {}
    
    def vgg(self):
        return []
    
    def voc(self):
        return None
    
    def yolo(self):
        return []
    
    def paperjs(self):
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
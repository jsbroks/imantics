from .styles import *


class Semantic:

    def __init__(self, id, metadata={}):
        self.id = id
        self.metadata = metadata
    
    def _coco(self):
        return {}
    
    def _vgg(self):
        return []
    
    def _voc(self):
        return None
    
    def _yolo(self):
        return []
    
    def _paperjs(self):
        return {}
    
    def export(self, style=COCO):

        return {
            COCO: self._coco(),
            VGG: self._vgg(),
            YOLO: self._yolo(),
            VOC: self._voc(),
            PAPERJS: self._paperjs()
        }.get(style)
    
    def save(self, file):
        pass
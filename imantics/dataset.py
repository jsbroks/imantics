class Dataset:
    
    @classmethod
    def from_coco(cls):
        pass
    
    def __init__(self, name, images, categories):
        self.name = name
        self.images = images
        self.categories = categories


__all__ = ["Dataset"]
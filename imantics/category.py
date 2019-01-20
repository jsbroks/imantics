from .basic import Semantic


class Category(Semantic):

    def __init__(self, name, parent=None, metadata={}, id=0):
        self.id = id
        self.name = name
        self.parent = None

        super().__init__(id, metadata)

    def _coco(self, include=True):

        category = {
            'id': self.id,
            'name': self.name,
            'supercategory': self.parent.name if self.parent else None,
            'metadata': self.metadata
        }

        if include:
            return {
                'categories': [category]
            }

        return category


__all__ = ["Category"]
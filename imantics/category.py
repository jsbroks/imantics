class Category:

    def __init__(self, name, parent=None, metadata={}, id=0):
        self.id = id
        self.name = name
        self.metadata = metadata
        self.parent = None

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
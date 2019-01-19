class Category:

    def __init__(self, name, parent=None, metadata={}):
        self.name = name
        self.metadata = metadata
        self.parent = None


__all__ = ["Category"]
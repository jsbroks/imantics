import numpy as np
import cv2


class Annotation:
        
    @classmethod
    def from_mask(cls, image, category, mask):
        return cls(image, category, mask=mask)
    
    @classmethod
    def from_bbox(cls, image, category, bbox):
        return cls(image, category, bbox=bbox)
    
    @classmethod
    def from_polygons(cls, image, category, polygons):
        return cls(image, category, polygons=polygons)

    def __init__(self, image, category, bbox=None, mask=None, polygons=None, id=0, color=None, metadata={}):
        
        assert isinstance(id, int), "id must be an integer"
        assert bbox or mask or polygons, "you must provide a mask, bbox or polygon"
        
        self.id = id
        self.image = image
        self.category = category
        
        self._c_bbox = BBox.create(bbox)
        self._c_mask = Mask.create(mask)
        self._c_polygons = Polygons.create(polygons)

        self._init_with_bbox = self._c_bbox is not None
        self._init_with_mask = self._c_mask is not None
        self._init_with_polygons = self._c_polygons is not None

        self.metadata = metadata

    @property
    def mask(self):
        if not self._c_mask:
            if self._init_with_polygons:
                self._c_mask = self.polygons.mask(width=self.image.width, height=self.image.height)
            else:
                self._c_mask = self.bbox.mask(width=self.image.width, height=self.image.height)
        
        return self._c_mask
    
    @property
    def array(self):
        return self.mask.array

    @property
    def polygons(self):

        if not self._c_polygons:
            if self._init_with_mask:
                self._c_polygons = self.mask.polygons()
            else:
                self._c_polygons = self.bbox.polygons()

        return self._c_polygons

    @property
    def bbox(self):
        if not self._c_bbox:   
            if self._init_with_polygons:
                self._c_bbox = self.polygons.bbox()
            else:
                self._c_bbox = self.mask.bbox()

        return self._c_bbox

    @property
    def area(self):
        return self.mask.area()

    @property
    def size(self):
        return self.image.size

    def __contains__(self, item):
        return self.mask.contains(item)

    def _vgg(self):
        pass

    def _coco(self, id, image, category):
        coco = {}
        coco['id'] = int(id)
        coco['image_id'] = image.id
        coco['width'] = image.width
        coco['height'] = image.height
        coco['category_id'] = category.id
        coco['area'] = self.area
        coco['segmentations'] = self.polygons
        coco['bbox'] = self.bbox.style(BBox.WIDTH_HEIGHT)
        coco['metadata'] = self.metadata
        return coco


class BBox:
    """
    Bounding Box Class
    """
    INSTANCE_TYPES = (np.ndarray, list, tuple)
    MIN_MAX = 'minmax'
    WIDTH_HEIGHT = 'widthheight'

    @classmethod
    def from_mask(cls, mask):
        return mask.bbox()
    
    @classmethod
    def from_polygons(cls, polygons):
        return polygons.bbox()
    
    @classmethod
    def create(cls, bbox):
        if isinstance(bbox, BBox.INSTANCE_TYPES):
            return BBox(bbox)
        
        if isinstance(bbox, BBox):
            return bbox
        
        return None
    
    @classmethod
    def empty(cls):
        return BBox((0, 0, 0, 0))

    _c_polygons = None
    _c_mask = None

    def __init__(self, bbox, style=None):
        """
        :param bbox:
        :param style: maxmin or widthheight
        """
        assert len(bbox) == 4

        self.style = style if style else BBox.MIN_MAX
        
        self._xmin = bbox[0]
        self._ymin = bbox[1]

        if self.style == self.MIN_MAX:
            self._xmax = bbox[2]
            self._ymax = bbox[3]
            self.width = self._xmax - self._xmin
            self.height = self._ymax - self._ymin

        if self.style == self.WIDTH_HEIGHT:
            self.width = bbox[2]
            self.height = bbox[3]
            self._xmax = self._xmin + self.width
            self._ymax = self._ymin + self.height
        
        self.area = self.width * self.height

    def bbox(self, style=None):
        style = style if style else self.style
        if style == self.MIN_MAX:
            return self._xmin, self._ymin, self._xmax, self._ymax
        return self._xmin, self._ymin, self.width, self.height

    def polygons(self):
        if not self._c_polygons:
            polygon = self.top_left + self.top_right \
                    + self.bottom_right + self.bottom_left
            return Polygons([polygon])
        return self._c_polygons

    def mask(self, width=None, height=None):
        if not self._c_mask:

            size = height, width if height and width else self.max_point[1], self.max_point[0]
            mask = np.zeros((height, width))
            mask[self.min_point[1]:self.max_point[1], self.min_point[0]:self.max_point[0]] = 1
                
            self._c_mask = Mask(mask)

        return self._c_mask

    def apply(self, image, color=None, thickness=2):
        color = color if color else (255, 0, 0)
        cv2.rectangle(image, self.min_point, self.max_point, color, thickness)

    @property
    def min_point(self):
        return self._xmin, self._ymin

    @property
    def max_point(self):
        return self._xmax, self._ymax
    
    @property
    def top_right(self):
        return self._xmax, self._ymin

    @property
    def top_left(self):
        return self._xmin, self._ymax
        
    @property
    def bottom_right(self):
        return self._xmax, self._ymax

    @property
    def bottom_left(self):
        return self._xmin, self._ymax
    
    @property
    def size(self):
        return self.width, self.height

    def _compute_size(self):
        self.width = self._xmax - self._xmin
        self.height = self._ymax - self._ymin

    def __getitem__(self, item):
        return self.bbox()[item]

    def __repr__(self):
        return repr(self.bbox())
    
    def __str__(self):
        return str(self.bbox())
    
    def __eq__(self, other):
        if isinstance(other, self.INSTANCE_TYPES):
            other = BBox(other)
        
        if isinstance(other, BBox):
            return np.array_equal(self.bbox(style=self.MIN_MAX), other.bbox(style=self.MIN_MAX))

        return False


class Polygons:
    
    INSTANCE_TYPES = (list, tuple)

    @classmethod
    def from_mask(cls, mask):
        return mask.polygons()

    @classmethod
    def from_bbox(cls, bbox, style=None):
        return bbox.polygons()
        
    @classmethod
    def create(cls, polygons):

        if isinstance(polygons, Polygons.INSTANCE_TYPES):
            return Polygons(polygons)
        
        if isinstance(polygons, Polygons):
            return polygons
            
        return None
    
    _c_bbox = None
    _c_mask = None
    _c_points = None

    def __init__(self, polygons):
        self.polygons = [np.array(polygon).flatten() for polygon in polygons]
    
    def mask(self, width=None, height=None):
        if not self._c_mask:

            size = height, width if height and width else self.bbox().max_point
            # Generate mask from polygons

            mask = np.zeros(size)
            mask = cv2.fillPoly(mask, self.points, 1)
            
            self._c_mask = Mask(mask)
            self._c_mask._c_polygons = self
        
        return self._c_mask

    def bbox(self):
        if not self._c_bbox:

            y_min = x_min = float('inf')
            y_max = x_max = float('-inf')

            for point_list in self.points:
                minx, miny = np.min(point_list, axis=0)
                maxx, maxy = np.max(point_list, axis=0)

                y_min = min(miny, y_min)
                x_min = min(minx, x_min)
                y_max = max(maxy, y_max)
                x_max = max(maxx, x_max)

            self._c_bbox = BBox((x_min, y_min, x_max, y_max))
            self._c_bbox._c_polygons = self

        return self._c_bbox

    def simplify(self):
        # TODO: Write simplification algotherm
        self._c_points = None

    @property
    def points(self):
        if not self._c_points:
            self._c_points = [
                np.array(point).reshape(-1, 2).round().astype(int)
                for point in self.polygons
            ]
        
        return self._c_points

    def __eq__(self, other):
        if isinstance(other, self.INSTANCE_TYPES):
            other = Polygons(other)
        
        if isinstance(other, Polygons):
            for i in range(len(self.polygons)):
                if not np.array_equal(self[i], other[i]):
                    return False
            return True
    
        return False
    
    def __getitem__(self, key):
        return self.polygons[key]

    def __repr__(self):
        return repr(self.polygons)


class Mask:
    """
    Mask class
    """

    INSTANCE_TYPES = (np.ndarray,)

    @classmethod
    def from_polygons(cls, polygons):
        return polygons.mask()
    
    @classmethod
    def from_bbox(cls, bbox):
        return bbox.mask()
            
    @classmethod
    def create(cls, mask):
        if isinstance(mask, Mask.INSTANCE_TYPES):
            return Mask(mask)
        
        if isinstance(mask, Mask):
            return mask
        
        return None

    _c_bbox = None
    _c_polygons = None

    def __init__(self, array):
        self.array = np.array(array, dtype=bool)
    
    def bbox(self):
        if not self._c_bbox:

            # Generate bbox from mask
            rows = np.any(self.array, axis=1)
            cols = np.any(self.array, axis=0)

            if not np.any(rows) or not np.any(cols):
                return BBox.empty()

            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            self._c_bbox = BBox((cmin, rmin, cmax, rmax))
            self._c_bbox._c_mask = self

        return self._c_bbox
    
    def polygons(self):
        if not self._c_polygons:

            # Generate polygons from mask
            mask = self.array.astype(np.uint8)
            mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
            polygons = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE, offset=(-1, -1))
            polygons = polygons[0] if len(polygons) == 2 else polygons[1]
            polygons = [polygon.flatten() for polygon in polygons]

            self._c_polygons = Polygons(polygons)
            self._c_polygons._c_mask = self

        return self._c_polygons
    
    def union(self, other):
        """
        Unites the array of the specified mask with this mask’s array and
        returns the result as a new mask.
        
        :param other: mask (or numpy array) to unite with
        :return: resulting mask
        """
        if isinstance(other, np.ndarray):
            other = Mask(other)
        
        return Mask(np.logical_or(self.array, other.array))
    
    def __add__(self, other):
        return self.union(other)

    def intersect(self, other):
        """
        Intersects the array of the specified mask with this masks’s array
        and returns the result as a new mask.

        :param other: mask (or numpy array) to intersect with
        :return: resulting mask
        """
        if isinstance(other, np.ndarray):
            other = Mask(other)

        return Mask(np.logical_and(self.array, other.array))
    
    def __mul__(self, other):
        return self.intersect(other)

    def iou(self, other):
        """
        Intersect over union value of the specified masks

        :param other: mask (or numpy array) to compute value with
        :return: resulting float value
        """
        i = self.intersect(other).sum()
        u = self.union(other).sum()
        
        if i == 0 or u == 0:
            return 0

        return i / float(u)
    
    def invert(self):
        return Mask(np.invert(self.array))
    
    def __invert__(self):
        return self.invert()
    
    def apply(self, image, color=None, alpha=0.5):        
        color = color if color else (255, 0, 0)
        for c in range(3):
            image[:, :, c] = np.where(
                self.array,
                image[:, :, c] * (1 - alpha) + alpha * color[c],
                image[:, :, c]
            )
        
        return image

    def subtract(self, other):
        """
        Subtracts the array of the specified mask from this masks’s array and
        returns the result as a new mask.

        :param other: mask (or numpy array) to subtract
        :retrn: resulting mask
        """
        if isinstance(other, np.ndarray):
            other = Mask(other)
        
        return self.intersect(other.invert())

    def __sub__(self, other):
        return self.subtract(other)

    def contains(self, item):
        """
        Checks whether a point (tuple), array or mask is within current mask.
        Note: Masks and arrays must be fully contained to return True
        
        :param item: object to check
        :return: boolean if item is contained
        """
        if isinstance(item, tuple):
            array = self.array
            for i in item:
                array = array[i]
            return array
        
        if isinstance(item, np.ndarray):
            item = Mask(item)

        if isinstance(item, Mask):
            return self.intersect(item).area() > 0
        
        return False

    def __contains__(self, item):
        return self.contains(item)
    
    def sum(self):
        return self.array.sum()

    def area(self):
        return self.sum()
    
    def __getitem__(self, key):
        return self.array[key]
    
    def __setitem__(self, key, value):
        self.array[key] = value
    
    def __eq__(self, other):
        if isinstance(other, (np.ndarray, list)):
            other = Mask(other)
        
        if isinstance(other, Mask):
            return np.array_equal(self.array, other.array)

        return False
    
    def __repr__(self):
        return repr(self.array)


__all__ = ["Annotation", "BBox", "Mask", "Polygons"]

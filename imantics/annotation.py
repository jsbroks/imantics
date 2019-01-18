import numpy as np
import cv2


class Annotation:

    _segments = []
    _bbox = []
    _mask = []

    image = None
    category = None
    metadata = {}

    def __init__(self, width=0, height=0, mask=None, segments=None, bbox=None):

        if width < 1 or height < 1:
            raise ValueError("Please provide a valid height and width of image")

        self._height = height
        self._width = width

        provided = False
        if mask:
            provided = True
            self.mask = mask

        elif segments:
            provided = True
            self.segments = segments

        elif bbox:
            provided = True
            self._bbox = bbox
            self._update_segments_from_bbox()
            self._update_mask_from_segments()

        if not provided:
            raise ValueError(
                "Please provided one of the follow, segments, bbox, or mask")

    @property
    def segments(self):

        if len(self._segments) == 0 or \
                (len(self._segments) == 1 and len(self._segments[0]) == 4):

            self._update_segments_from_bbox()

        return self._segments

    @segments.setter
    def segments(self, value):
        self._segments = value

        self._update_mask_from_segments()
        self._update_bbox_from_mask()

    @property
    def mask(self):
        if np.shape(self._mask) == 0:
            self._update_mask_from_segments()

        return self._mask

    @mask.setter
    def mask(self, value):

        assert value.shape == (
            self._height, self._width), "Mask shape needs to match height and width"

        self._mask = value

        self._update_bbox_from_mask()
        self._update_segments_from_mask()

    @property
    def bbox(self):
        if not self._bbox:
            self._update_bbox_from_mask()

        return self._bbox

    @bbox.setter
    def bbox(self, value):
        self._bbox = value

    def area(self):
        return self.mask.sum()

    def add(self, mask):
        pass

    def subtract(self, mask):
        pass

    def simplify(self, tolerance):
        """ TODO
        Using the Douglas Peucker algorithm to simplify the segmentation

        :param tolerance: amount of simplification that occurs (the smaller, the less simplification)
        """
        pass

    def _update_mask_from_segments(self):
        mask = np.zeros((self._height, self._width))

        points = [
            np.array(point).reshape(-1, 2).round().astype(int)
            for point in self._segments
        ]

        self._mask = cv2.fillPoly(mask, points, 1)

    def _update_bbox_from_mask(self):

        rows = np.any(self._mask, axis=1)
        cols = np.any(self._mask, axis=0)

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        self._bbox = (cmin, rmin, cmax, rmax)

    def _update_segments_from_mask(self):
        pass

    def _update_segments_from_bbox(self):

        width = self.bbox[2] - self.bbox[0]

        segment = [
            self.bbox[0], self.bbox[1],
            self.bbox[0] + width, self.bbox[1],
            self.bbox[2], self.bbox[3],
            self.bbox[2] - width, self.bbox[3]
        ]

        self._segments = [segment]

    def __contains__(self, item):
        return True


class BBox:
    """
    Bounding Box Class
    """

    MIN_MAX = 'minmax'
    WIDTH_HEIGHT = 'widthheight'

    @classmethod
    def from_mask(cls, mask):
        """
        Returns bounding box class of a mask

        :param mask: Numpy binary mask
        :return: BBox class
        """
        if isinstance(mask, Mask):
            return mask.bbox

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return cls((cmin, rmin, cmax, rmax))
    
    @classmethod
    def from_segments(cls, segment):
        pass

    def __init__(self, bbox, style='minmax'):
        """
        :param bbox:
        :param style: maxmin or widthheight
        """
        assert len(bbox) == 4

        self.style = style

        self._xmin = bbox[0]
        self._ymin = bbox[1]

        if style == self.MIN_MAX:
            self._xmax = bbox[2]
            self._ymax = bbox[3]
            self._compute_size()

        if style == self.WIDTH_HEIGHT:
            self.width = bbox[2]
            self.height = bbox[3]
            self._compute_max_point()

        self.area = self.width * self.height
        self._compute_segment()

    @property
    def bbox(self):
        if self.style == self.MIN_MAX:
            return self._xmin, self._ymin, self._xmax, self._ymax
        return self._xmin, self._ymin, self.width, self.height

    @property
    def min_point(self):
        return self._xmin, self._ymin

    @property
    def max_point(self):
        return self._xmax, self._ymax

    @property
    def size(self):
        return self.width, self.height
    
    def _compute_segment(self):

        self.segment = [[
            self._xmin, self._ymin,
            self._xmin + self.width, self._ymin,
            self._xmax, self._ymax,
            self._xmax - self.width, self._ymax
        ]]

    def __getitem__(self, item):
        return self.bbox[item]

    def __repr__(self):
        return repr(self.bbox)

    def _compute_max_point(self):
        self._xmax = self._xmin + self.width
        self._ymax = self._ymin + self.height

    def _compute_size(self):
        self.width = self._xmax - self._xmin
        self.height = self._ymax - self._ymin


class Segments:

    def __init__(self):
        pass


class Mask:
    """
    Mask class
    """
    _c_bbox = None
    _c_segments = None

    def __init__(self, array):
        self.array = np.array(array, dtype=bool)

    def bbox(self):
        if not self._c_bbox:
            self._c_bbox = BBox.from_mask(self.array)
        return self._c_bbox
    
    def segments(self):
        if not self._c_segments:
            self._c_segments = Segments.from_mask(self.array)
        return self._c_segments
    
    def union(self, other):
        """
        Unites the array of the specified mask with this mask’s array and returns the result as a new mask.
        
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
        Intersects the array of the specified mask with this masks’s array and returns the result as a new mask.

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

    def subtract(self, other):
        """
        Subtracts the array of the specified mask from this masks’s array and returns the result as a new mask.

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
            print(self.array, other.array)
            return np.array_equal(self.array, other.array)

        return False

__all__ = ["Annotation", "BBox", "Mask"]

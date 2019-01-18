import pytest
import numpy as np
from imantics import Mask

test_intersect = [
    # array a, array b, expect intersected array
    ([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]),
    ([1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]),
    ([1, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]),
    ([1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1])
]

test_union = [
    # array a, array b, expect union array
    ([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]),
    ([1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 1]),
    ([1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]),
    ([1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1])
]

test_iou = [
    # array a, array b, expected iou
    ([0, 0, 0, 0], [0, 0, 0, 0], 0),
    ([0, 0, 1, 1], [1, 1, 0, 0], 0),
    ([1, 1, 0, 0], [1, 0, 0, 0], 1/2),
    ([1, 1, 1, 1], [1, 1, 1, 1], 1)
]

test_invert = [
    # array, inverted
    ([0, 0, 0, 0], [1, 1, 1, 1]),
    ([0, 1, 1, 0], [1, 0, 0, 1]),
    ([0, 0, 0, 0], [1, 1, 1, 1])

]

test_subtract = [
    # array a, array b, expected array
    ([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]),
    ([1, 1, 1, 1], [1, 0, 0, 0], [0, 1, 1, 1]),
    ([1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0])
]

test_area = [
    # array, expected area
    ([0, 0, 0, 0], 0),
    ([1, 1, 0, 0], 2),
    ([1, 1, 1, 1], 4)
]

test_contains_point = [
    # array, point, expected contains
    ([0, 1, 1, 0], (1,), True),
    ([0, 1, 1, 0], (3,), False)  
]

test_contains_mask = [
     # array a, array b, expected contains
    ([0, 0, 0, 0], [0, 1, 0, 0], False),
    ([1, 1, 1, 1], [0, 1, 0, 0], True),
    ([1, 1, 1, 1], [0, 0, 0, 0], False),   
    ([1, 1, 1, 1], [1, 1, 1, 1], True),
    ([0, 0, 0, 0], [0, 0, 0, 0], False)    
]


class TestMaskInitalization:

    def test_from_segments(self):
        pass
    
    def test_from_bbox(self):
        pass


class TestMaskConversion:

    def test_to_bbox(self):
        pass
    
    def test_to_segments(self):
        pass


class TestMaskComputations:

    @pytest.mark.parametrize("array_a,array_b,e_union", test_union)
    def test_union(self, array_a, array_b, e_union):
        mask_a = Mask(array_a)
        mask_b = Mask(array_b)
        
        assert mask_a.union(mask_b) == e_union
        assert mask_b.union(mask_a) == e_union 
        assert mask_a + mask_b == e_union
        assert mask_a + np.array(array_b) == e_union
    
    @pytest.mark.parametrize("array_a,array_b,e_intersect", test_intersect)
    def test_intersect(self, array_a, array_b, e_intersect):
        mask_a = Mask(array_a)
        mask_b = Mask(array_b)
        
        assert mask_a.intersect(mask_b) == e_intersect
        assert mask_b.intersect(mask_a) == e_intersect 
        assert mask_a * mask_b == e_intersect
        assert mask_a * np.array(array_b) == e_intersect

    @pytest.mark.parametrize("array_a,array_b,e_iou", test_iou)
    def test_iou(self, array_a, array_b, e_iou):
        mask_a = Mask(array_a)
        mask_b =  Mask(array_b)

        assert mask_a.iou(mask_b) == e_iou
        assert mask_b.iou(mask_a) == e_iou

    @pytest.mark.parametrize("array,e_invert", test_invert)
    def test_invert(self, array, e_invert):
        mask = Mask(array)

        assert ~mask == e_invert
        assert mask.invert() == e_invert

    @pytest.mark.parametrize("array_a,array_b,e_subtract", test_subtract)
    def test_subtract(self, array_a, array_b, e_subtract):
        mask_a = Mask(array_a)
        mask_b = Mask(array_b)

        assert mask_a.subtract(mask_b) == e_subtract
        assert mask_a - mask_b == e_subtract
        assert mask_a - np.array(array_b) == e_subtract

    @pytest.mark.parametrize("array,point,e_contains", test_contains_point)
    def test_contains_point(self, array, point, e_contains):
        mask = Mask(array)

        assert mask.contains(point) == e_contains
        assert (point in mask) == e_contains

    @pytest.mark.parametrize("array_a,array_b,e_contains", test_contains_mask)
    def test_contains_mask(self, array_a, array_b, e_contains):
        mask_a = Mask(array_a)
        mask_b = Mask(array_b)
        array_b = np.array(array_b)

        assert mask_a.contains(mask_b) == e_contains
        assert (mask_b in mask_a) == e_contains
        assert mask_a.contains(array_b) == e_contains
        assert (array_b in mask_a) == e_contains




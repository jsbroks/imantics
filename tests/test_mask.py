import pytest
import numpy as np
from imantics import Mask

test_iou = [
    # array a, array b, expected iou
    ([0, 0, 0, 0], [0, 0, 0, 0], 0),
    ([0, 0, 1, 1], [1, 1, 0, 0], 0),
    ([1, 1, 0, 0], [1, 0, 0, 0], 1/2),
    ([1, 1, 1, 1], [1, 1, 1, 1], 1)
]


class TestMaskComputations:

    @pytest.mark.parametrize("array_a,array_b,e_iou", test_iou)
    def test_iou(self, array_a, array_b, e_iou):
        mask_a = Mask(array_a)
        mask_b =  Mask(array_b)

        print(mask_a)
        print(mask_b)

        assert mask_a.iou(mask_b) == e_iou
        assert mask_b.iou(mask_a) == e_iou

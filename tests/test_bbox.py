
import pytest
from imantics import BBox


test_shape = [
    # bbox, style, expected shape
    ([0, 0, 10, 15], BBox.MIN_MAX, (10, 15)),
    ([5, 5, 10, 15], BBox.MIN_MAX, (5, 10)),
    ([0, 0, 10, 15], BBox.WIDTH_HEIGHT, (10, 15)),
    ([5, 5, 10, 15], BBox.WIDTH_HEIGHT, (10, 15))
]

test_points = [
    # bbox, style, expected min point, expected max point
    ([0, 0, 10, 15], BBox.MIN_MAX, (0, 0), (10, 15)),
    ([5, 5, 10, 15], BBox.MIN_MAX, (5, 5), (10, 15)),
    ([0, 0, 10, 15], BBox.WIDTH_HEIGHT, (0, 0), (10, 15)),
    ([5, 5, 10, 15], BBox.WIDTH_HEIGHT, (5, 5), (15, 20))  
]

test_area = [
    # bbox, style, expected area
    ([0, 0, 10, 15], BBox.MIN_MAX, 150),
    ([5, 5, 10, 15], BBox.MIN_MAX, 50),
    ([0, 0, 10, 15], BBox.WIDTH_HEIGHT, 150),
    ([5, 5, 10, 15], BBox.WIDTH_HEIGHT, 150)  
]


class TestBBoxConstants:
    def test_styles(self):
        assert BBox.MIN_MAX == "minmax"
        assert BBox.WIDTH_HEIGHT == "widthheight"

    def test_default_bbox_style(self):
        sut = BBox([0, 0, 0, 0])
        assert sut.style == BBox.MIN_MAX


class TestBBoxMeasurements:

    @pytest.mark.parametrize("bbox,style,e_shape", test_shape)
    def test_shape(self, bbox, style, e_shape):
        sut = BBox(bbox, style=style)

        assert sut.size == e_shape
        assert sut.width == e_shape[0]
        assert sut.height == e_shape[1]
    
    @pytest.mark.parametrize("bbox,style,e_min,e_max", test_points)
    def test_points(self, bbox, style, e_min, e_max):
        sut = BBox(bbox, style=style)

        assert sut.min_point == e_min
        assert sut.max_point == e_max
    
    @pytest.mark.parametrize("bbox,style,e_area", test_area)
    def test_area(self, bbox, style, e_area):
        sut = BBox(bbox, style=style)

        assert sut.area == e_area

    def test_segmentation(self):
        pass


class TestBBoxStyle:
    
    def test_style_change(self):
        pass


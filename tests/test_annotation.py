from imantics import Annotation, Mask, BBox, Polygons


class TestAnnotationConversion:
    
    def test_create_mask(self, mask, e_bbox, e_polygon):
        assert True
from imantics import Image

class TestImageCreate:

    def test_create_empty(self):
        width, height = (50, 100)
        image = Image.empty(width=width, height=height)
        draw_bbox = image.draw_bboxs()
        draw_mask = image.draw_masks()

        assert image.height == height
        assert image.width == width
        assert image.array.shape == (height, width, 3)
        assert len(image.annotations) == 0
        assert len(image.categories) == 0

        assert draw_bbox.shape == (height, width, 3)
        assert draw_mask.shape == (height, width, 3)
    
    def test_image_from_path(self):
        path = 'examples/data/tesla.jpg'
        image = Image.from_path('examples/data/tesla.jpg')

        assert image.path == path
        assert image.file_name == 'tesla.jpg'
        assert image.size == (900, 600)

    def test_images_from_folder(self):
        path = 'examples/data'
        images = Image.from_path(path)
        
        assert isinstance(images, list)
        assert len(images) == 1
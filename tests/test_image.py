from imantics import Image

class TestImageCreate:

    def test_create_empty(self):
        width, height = (50, 100)
        image = Image.empty(width=width, height=height)
        draw = image.draw()

        assert image.height == height
        assert image.width == width
        assert len(image.annotations) == 0
        assert len(image.categories) == 0

        assert draw.shape == (height, width, 3)
    
    def test_image_from_path(self):
        path = 'examples/data/coco_example/tesla.jpg'
        image = Image.from_path('examples/data/coco_example/tesla.jpg')

        assert image.path == path
        assert image.file_name == 'tesla.jpg'
        assert image.size == (900, 600)

    def test_images_from_folder(self):
        path = 'examples/data/coco_example'
        images = Image.from_path(path)
        
        assert isinstance(images, list)
        assert len(images) == 1

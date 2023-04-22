import unittest
from preprocessing_and_training import *

class TestFunctions(unittest.TestCase):
    def test_predict(self):
        model_path = 'jordan_classifier.pkl'
        learn = load_learner(model_path)

        # high-quality product photo
        filename = 'jordan_2.jpg'
        img_path = f"uploads/{filename}"
        img = Image.open(img_path)
        img = img.crop_pad((192, 192))
        label, index, probs = learn.predict(img)
        self.assertEqual(label, 'jordan 2')

        # photo taken of personal shoe from iphone
        filename = 'jordan_1.jpg'
        img_path = f"uploads/{filename}"
        img = Image.open(img_path)
        img = img.crop_pad((192, 192))
        label, index, probs = learn.predict(img)
        self.assertEqual(label, 'jordan 1')

        # on-foot picture taken from internet
        filename = 'jordan_5.jpg'
        img_path = f"uploads/{filename}"
        img = Image.open(img_path)
        img = img.crop_pad((192, 192))
        label, index, probs = learn.predict(img)
        self.assertEqual(label, 'jordan 5')

    def test_is_readable_image_pass(self):
        filename = 'jordan_1.jpg'
        img_path = f"uploads/{filename}"

        # Test whether the image file is readable
        self.assertTrue(is_readable_image(img_path))

    def test_is_readable_image_fail(self):
        filename = 'jordan_1.HEIC'
        img_path = f"uploads/{filename}"

        # Test whether the image file is readable
        self.assertFalse(is_readable_image(img_path))


if __name__ == '__main__':
    unittest.main()

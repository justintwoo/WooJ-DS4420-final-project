import unittest
from jordan_image_recognition import *

class TestFunctions(unittest.TestCase):
    def test_predict(self):
        model_path = 'jordan_classifier.pkl'
        learn = load_learner(model_path)

        # high-quality product photo
        filename = 'jordan_2.jpg'
        img_path = f"uploads/{filename}"
        img = Image.open(img_path)
        img = np.array(img)
        label, index, probs = learn.predict(img)

        self.assertEqual(label, 'jordan 2')

        # photo taken of personal shoe from iphone
        img_path = f"uploads/{filename}"
        filename = 'jordan_1.jpg'
        img = Image.open(img_path)
        img = np.array(img)
        label, index, probs = learn.predict(img)

        self.assertEqual(label, 'jordan 1')

        # on-foot picture taken from internet
        img_path = f"uploads/{filename}"
        filename = 'jordan_5.jpg'
        img = Image.open(img_path)
        img = np.array(img)
        label, index, probs = learn.predict(img)

        self.assertEqual(label, 'jordan 5')



if __name__ == '__main__':
    unittest.main()

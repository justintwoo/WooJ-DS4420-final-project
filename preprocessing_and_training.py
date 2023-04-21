"""
Project: Jordan Sneaker Recognition Classification Model
Author: Justin Woo
"""
import os
import cv2
import matplotlib.pyplot as plt
import warnings
# temporarily filter out all warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
from duckduckgo_search import ddg_images
from fastai.vision.all import *
from pathlib import Path
from time import sleep
import PIL.Image as PILImage
from functools import partial

def search_images(term, max_images):
    """
    Search for images using DuckDuckGo search engine API.

    :param term: str, search term
    :param max_images: int, maximum number of images to return (default=100)
    :return: L, a list of image URLs
    """
    print(f"Searching for '{term}'")
    return L(ddg_images(term, max_results=max_images)).itemgot('image')

def get_filtered_image_files(path):
    """
    Get list of image files from a given directory, filtering out any that are not readable.

    :param path: str or Path, path to directory containing images
    :return: list, a list of readable image file paths
    """
    files = get_image_files(path)
    return [f for f in files if is_readable_image(f)]

def is_readable_image(file):
    """
    Verify whether an image file is readable.

    :param file: str or Path, file path to image file
    :return: bool, True if image is readable, False otherwise
    """
    try:
        PILImage.open(file).verify()
        return True
    except:
        return False

def download_training_data(search_terms, num_images = 100, path_name='jordans', max_size=400):
    """
    Download and save images to a specified directory based on a list of search terms.

    :param search_terms: list, a list of search terms (e.g. 'jordan 1', 'jordan 2', etc.)
    :param path_name: str or Path, name of directory to store downloaded images (default='jordans')
    :param max_size: int, maximum width or height of the downloaded images in pixels (default=400)
    :return: None
    """
    path = Path(path_name)

    for term in search_terms:
        dest = (path / term)
        dest.mkdir(exist_ok=True, parents=True)
        existing_files = get_filtered_image_files(dest)
        for file in existing_files:
            file.unlink()
        download_images(dest, urls=search_images(term, max_images=num_images))
        # sleep(10)  # Pause between searches to avoid over-loading server
        resize_images(path / term, max_size=max_size, dest=path / term)

def train_model(path = Path('jordans')):
    """
    Train a model using downloaded images stored in a specified directory.

    :param path: str or Path, path to directory containing downloaded images (default='jordans')
    :return: Learner, a fastai Learner object containing the trained model
    """
    failed = verify_images(get_image_files(path))
    failed.map(Path.unlink)
    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_filtered_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method='squish')]
    ).dataloaders(path, bs=32)

    # Set the path to the saved model
    learn = vision_learner(dls, resnet18, metrics=error_rate)

    learn.fine_tune(3)

    # Save the trained model
    model_path = Path('jordan_classifier.pkl')
    learn.export(model_path)

    return learn

def main():
        max_images = input('how many pictures per class for training?\n')

        search_terms = [f'jordan {i}' for i in range(1, 15)]

        # Collect and download images for training
        download_training_data(search_terms)

        # Specify path to jordans directory
        path = Path('jordans')

        # Remove images that fail verification
        failed = verify_images(get_image_files(path))
        failed.map(Path.unlink)

        # Train the model from downloaded data
        learn = train_model()

if __name__ == '__main__':
    main()





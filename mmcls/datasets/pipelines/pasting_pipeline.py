import os
import tarfile
import time
from glob import glob
from random import randint, choice

import numpy as np
from PIL import Image

from ..builder import PIPELINES


@PIPELINES.register_module()
class PastingPipeline(object):
    def __init__(self, cell_path, untar_path):
        self.cell_path = cell_path
        self.untar_path = untar_path

        # Untar if needed
        if self.cell_path.endswith('.tar'):
            self.untar_path = untar_to_dst(self.untar_path, self.cell_path)
            dataset_dir = self.cell_path.split('/')[-1].split('.')[0]
            self.cell_path = os.path.join(self.untar_path, dataset_dir)
            print(self.cell_path)

        # Collect the cells
        cell_query = os.path.join(self.cell_path, "*/*.jpg")
        self.cell_filepaths = self.get_cell_filepaths(cell_query)

    def get_cell_filepaths(self, cell_query):
        # Get all the available images
        filepaths = [f for f in glob(cell_query, recursive=True)]

        # Split in positive/negative
        cells_filepaths = {
            'positives': list(filter(lambda x: 'positives' in x, filepaths)),
            'negatives': list(filter(lambda x: 'negatives' in x, filepaths)),
        }
        return cells_filepaths

    def __call__(self, results):
        image = results['img']
        label = results['gt_label']

        # Load the image
        image_paste = image.copy()

        # Load the cell image
        str_label = 'positives' if label == np.array(1) else 'negatives'
        cell_path = choice(self.cell_filepaths[str_label])
        cell_image = np.array(Image.open(cell_path))

        # Compute the location of the pasting site
        cell_h, cell_w, _ = cell_image.shape
        image_h, image_w, _ = image.shape
        if cell_h <= image_h:
            max_r = image_h - cell_h
        else:
            max_r = 0
            diff = cell_h - image_h
            low = diff // 2
            high = diff - low
            cell_image = cell_image[low: -high]
        if cell_w <= image_w:
            max_c = image_w - cell_w
        else:
            max_c = 0
            diff = cell_w - image_w
            left = diff // 2
            right = diff - left
            cell_image = cell_image[:, left: -right]

        paste_r = randint(0, max_r)
        paste_c = randint(0, max_c)

        # Paste
        # lam = np.random.uniform(0., 1.0)
        # Paste in rgb
        # image_paste[paste_r: paste_r + cell_h, paste_c: paste_c + cell_w] = \
        #     lam * image[paste_r: paste_r + cell_h, paste_c: paste_c + cell_w] + (1. - lam) * cell_image
        image_paste[paste_r: paste_r + cell_h, paste_c: paste_c + cell_w] = cell_image

        results['img'] = image_paste
        return results

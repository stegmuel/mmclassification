import os
import tarfile
import time
from glob import glob
from random import randint, choice
from einops import rearrange
from scipy.signal import fftconvolve
import cv2 as cv

import numpy as np
from PIL import Image

from ..builder import PIPELINES
import torchvision.transforms as pth_transforms


def untar_to_dst(untar_path, src):
    assert (untar_path != "")

    if untar_path[0] == '$':
        untar_path = os.environ[untar_path[1:]]
    start_copy_time = time.time()

    with tarfile.open(src, 'r') as f:
        f.extractall(untar_path)
    print('Time taken for untar:', time.time() - start_copy_time)

    # Wait
    time.sleep(5)
    return untar_path


@PIPELINES.register_module()
class PastingPipeline(object):
    def __init__(self, cell_path, untar_path, random_paste=True, paste_mode='paste', paste_negatives=True,
                 augment_cells=False):
        self.cell_path = cell_path
        self.untar_path = untar_path
        self.random_paste = random_paste
        self.paste_mode = paste_mode
        self.paste_negatives = paste_negatives
        self.augment_cells = augment_cells
        if augment_cells:
            self.cell_transform = pth_transforms.Compose([
                pth_transforms.RandomHorizontalFlip(),
                pth_transforms.RandomHorizontalFlip(),
                pth_transforms.RandomApply(
                    [pth_transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.5
                ),
                pth_transforms.RandomApply(
                    [pth_transforms.GaussianBlur(kernel_size=7)],
                    p=0.2
                ),
                pth_transforms.RandomGrayscale(p=0.2),
            ])

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

        # Do nothing if we don't paste on negative images and the label is 0
        if label == np.array(0) and not self.paste_negatives:
            return results

        # Load the image
        im_pasted = np.array(image.copy())

        # Load the cell image
        str_label = 'positives' if label == np.array(1) else 'negatives'
        cell_path = choice(self.cell_filepaths[str_label])
        cell_image = Image.open(cell_path)
        if self.augment_cells:
            cell_image = self.cell_transform(cell_image)

        cell_image = np.array(cell_image)

        # Crop the cell image if needed
        cell_h, cell_w, _ = cell_image.shape
        image_h, image_w, _ = im_pasted.shape
        if cell_h > image_h:
            diff = cell_h - image_h
            low = diff // 2
            high = diff - low
            cell_image = cell_image[low: -high]
            cell_h = image_h
        if cell_w > image_w:
            diff = cell_w - image_w
            left = diff // 2
            right = diff - left
            cell_image = cell_image[:, left: -right]
            cell_w = image_w

        # Compute the location of the pasting site
        if self.random_paste:
            max_r = image_h - cell_h
            max_c = image_w - cell_w
            center_y = randint(0, max_r)
            center_x = randint(0, max_c)
            center = (center_x + cell_image.shape[1] // 2, center_y + cell_image.shape[0] // 2)
        else:
            kernel = np.ones(cell_image.shape) / cell_image.size
            convolved = fftconvolve(im_pasted, kernel, mode='valid').squeeze()
            h, w = convolved.shape[:2]
            center = rearrange(convolved, 'h w -> (h w)').argmax()
            center_x = center % w
            center_y = center // w
            center = (center_x + cell_image.shape[1] // 2, center_y + cell_image.shape[0] // 2)

        # Paste
        if self.paste_mode == 'paste':
            im_pasted[center_y: center_y + cell_h, center_x: center_x + cell_w] = cell_image
        elif self.paste_mode == 'blend':
            lam = np.random.uniform(0., 1.0)
            im_pasted[center_y: center_y + cell_h, center_x: center_x + cell_w] = \
                lam * im_pasted[center_y: center_y + cell_h, center_x: center_x + cell_w] + (1. - lam) * cell_image
        else:
            im_pasted = cv.seamlessClone(cell_image, im_pasted, 255 * np.ones_like(cell_image), center, cv.NORMAL_CLONE)
        results['img'] = im_pasted
        return results

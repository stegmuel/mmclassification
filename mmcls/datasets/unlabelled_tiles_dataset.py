import os
import pickle
from glob import glob
from random import sample

import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS
from .utils import untar_to_dst


@DATASETS.register_module()
class UnlabelledTilesDataset(BaseDataset):

    def __init__(self,
                 data_prefix,
                 pipeline,
                 classes=None,
                 ann_file=None,
                 test_mode=False,
                 phase='train',
                 untar_path=None,
                 samples_per_class=1000,
                 ):
        self.phase = phase
        self.untar_path = untar_path
        self.samples_per_class = samples_per_class
        super(UnlabelledTilesDataset, self).__init__(data_prefix, pipeline, classes, ann_file, test_mode)

    def load_annotations(self):
        assert isinstance(self.data_prefix, str)
        assert isinstance(self.ann_file, str)

        # Untar if needed
        if self.data_prefix.endswith('.tar'):
            self.untar_path, self.data_prefix = untar_to_dst(self.untar_path, self.data_prefix)
            print(self.data_prefix)

        # Load the annotation file
        with open(self.ann_file, 'rb') as f:
            annotations = set(pickle.load(f)[self.phase])

        data_infos = []
        for dir in ['negatives', 'positives']:
            # Define the query
            query = os.path.join(self.data_prefix, f"{dir}/*/*.jpg")

            # Sample
            files = glob(query)

            # Filter out samples from the wrong phase
            files = list(filter(lambda f: '_'.join(f.split('/')[-1].split('_')[:-2]) in annotations, files))

            # Randomly sample the desired number of samples
            files = sample(files, self.samples_per_class)

            # Store
            infos = {
                'img_prefix': self.samples_per_class * [self.data_prefix],
                'img_info': [{'filename': f} for f in files],
                'gt_label': self.samples_per_class * [np.array(1 if dir == 'positives' else 0, dtype=np.int64)]
            }
            infos = [dict(zip(infos, v)) for v in zip(*infos.values())]
            data_infos.extend(infos)
        return data_infos

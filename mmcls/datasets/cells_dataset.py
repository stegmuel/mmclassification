import mmcv
import numpy as np
import os
import pickle
import tarfile
import time
import torch
from glob import glob

from .base_dataset import BaseDataset
from .builder import DATASETS


def untar_to_dst(untar_path, src):
    assert (untar_path != "")

    if untar_path[0] == '$':
        untar_path = os.environ[untar_path[1:]]
    start_copy_time = time.time()

    with tarfile.open(src, 'r') as f:
        f.extractall(untar_path)
    print('Time taken for untar:', time.time() - start_copy_time)

    time.sleep(5)


@DATASETS.register_module()
class CellsDataset(BaseDataset):

    def __init__(self,
                 data_prefix,
                 pipeline,
                 classes=None,
                 ann_file=None,
                 test_mode=False,
                 untar_path=None,
                 ):
        self.untar_path = untar_path
        super(CellsDataset, self).__init__(data_prefix, pipeline, classes, ann_file, test_mode)

    def load_annotations(self):
        assert isinstance(self.data_prefix, str)

        # Untar if needed
        if self.data_prefix.endswith('.tar'):
            untar_to_dst(self.untar_path, self.data_prefix)
            dataset_dir = self.data_prefix.split('/')[-1].split('.')[0]
            self.data_prefix = os.path.join(self.untar_path, dataset_dir)
            print(self.data_prefix)

        data_infos = []
        query = os.path.join(self.data_prefix, '*/*.jpg')
        for f in glob(query):
            label_dir = f.split('/')[-2]

            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': f}
            gt_label = 1 if label_dir == 'positives' else 0
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos

import numpy as np
import os
from glob import glob

from .base_dataset import BaseDataset
from .builder import DATASETS
from utils import untar_to_dst


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
            self.untar_path, self.data_prefix = untar_to_dst(self.untar_path, self.data_prefix)
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

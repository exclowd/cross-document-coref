from typing import Tuple

import os

from lightning import LightningDataModule
from prepare_dataset import prepare_dataset


class ECBPlusDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 train_test_val_split: Tuple[int, int, int],
                 batch_size: int = 64,
                 num_workers: int = 8,
                 ):
        super().__init__(self)
        self.raw_data_dir = data_dir
        self.data_dir: str = f"{data_dir}_prep"
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        # TODO make this download data only if needed
        if not os.path.exists(self.raw_data_dir):
            raise Exception(f"{self.raw_data_dir} not found")
        prepare_dataset(self.raw_data_dir, self.data_dir)

    def setup(self, stage: str):
        if stage == 'fit':
            pass
        if stage == 'test':
            pass
        if stage == 'predict':
            pass

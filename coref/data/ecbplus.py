import json
from typing import Dict, List, Optional, Tuple

import os
from lightning import LightningDataModule
from transformers import AutoTokenizer

from torch.utils.data import Dataset
from prepare_dataset import prepare_dataset


class ECBPlusDataset(Dataset):
    def __init__(self,
                 docs: Dict[Dict],
                 mentions: Dict[Dict],
                 tokenizer: AutoTokenizer,
                 predicted_topics: Optional[List[str]] = None,
                 segment_window: int = 128,
                 use_subtopic: bool = False,
                 ):
        self.docs = docs
        self.mentions = mentions
        self.tokenizer = tokenizer

        self.segment_window = segment_window

        self.topic_list = []
        self.topic2docs = {}

        self.label

        for m in mentions:
            doc_id = m['doc_id']
            




class ECBPlusDataModule(LightningDataModule):
    def __init__(self,
                 mention_type: str,
                 data_dir: str,
                 tokenizer: AutoTokenizer,
                 topics: Dict[str, List],
                 use_gold_mentions: bool = True,
                 use_subtopic: bool = False,
                 batch_size: int = 64,
                 num_workers: int = 8,
                 ):
        super().__init__(self)
        self.data_dir = data_dir
        self.mention_type = mention_type
        self.tokenizer = tokenizer

        self.train_dataset: Optional[ECBPlusDataset] = None

    def prepare_data(self) -> None:
        # cannot assign
        if not os.path.exists(self.raw_data_dir):
            raise Exception(f"{self.raw_data_dir} not found")
        prepare_dataset(self.raw_data_dir, self.data_dir)

    def setup(self, stage: str):
        docs_path = os.path.join(self.data_dir, f'{stage}_docs.json')
        mention_path = os.path.join(
            self.data_dir, f'{stage}_{self.mention_type}.json')
        if stage == 'fit':
            with open(mention_path, 'r') as f:
                mentions = json.load(f)
            with open(docs_path, 'r') as f:
                docs = json.load(f)
            self.train_dataset = ECBPlusDataset(docs, mentions, self.tokenizer)
        if stage == 'test':
            pass
        if stage == 'predict':
            pass

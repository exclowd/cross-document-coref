import torch
import numpy as np


def fix_seed(random_seed: int):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Metrics:
    def __init__(self):
        self.predictions = None
        self.labels = None
        self.tp = None
        self.tn = None
        self.fp = None
        self.fn = None
        self.tp_num = None
        self.tn_num = None
        self.fp_num = None
        self.fn_num = None
        self.total = None
        self.precision = None
        self.recall = None

    def compute(self, predictions, labels):
        # predictions: [batch_size, seq_len]
        # gold_labels: [batch_size, seq_len]
        # return: [batch_size, seq_len]
        self.predictions = predictions
        self.labels = labels
        self.tp = (predictions == 1) * (labels == 1)

        self.tp_num = torch.nonzero(self.tp).squeeze().shape[0]
        self.tn = (predictions != 1) * (labels != 1)
        self.tn_num = torch.nonzero(self.tn).squeeze().shape[0]
        self.fp = (predictions == 1) * (labels != 1)
        self.fp_num = torch.nonzero(self.fp).squeeze().shape[0]
        self.fn = (predictions != 1) * (labels == 1)
        self.fn_num = torch.nonzero(self.fn).squeeze().shape[0]
        self.total = len(labels)

        self.precision = self.tp_num / \
            (self.tp_num + self.fp_num) if self.tp_num + self.fp_num != 0 else 0
        self.recall = self.tp_num / \
            (self.tp_num + self.fn_num) if self.tp_num + self.fn_num != 0 else 0

    def get_fp(self):
        return torch.nonzero(self.fp).squeeze()

    def get_tp(self):
        return torch.nonzero(self.tp).squeeze()

    def get_tn(self):
        return torch.nonzero(self.tn).squeeze()

    def get_fn(self):
        return torch.nonzero(self.fn).squeeze()

    def get_accuracy(self):
        return (self.tp_num + self.tn_num) / self.total

    def get_precision(self):
        return self.precision

    def get_recall(self):
        return self.recall

    def get_f1(self):
        return 2 * self.precision * self.recall / (self.precision + self.recall) if (self.precision + self.recall) > 0 else 0

    def print(self) -> str:
        print(f'Accuracy: {self.get_accuracy()}')
        print(f'Precision: {self.get_precision()}')
        print(f'Recall: {self.get_recall()}')
        print(f'F1: {self.get_f1()}')
        return ''
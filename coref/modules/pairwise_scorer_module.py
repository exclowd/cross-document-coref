from lightning import LightningModule
from torchmetrics import MeanMetric

import torch
from coref.models import PairwiseScorer, SpanScorer, SpanEmbedder
from coref.utils import Metrics


class PairwiseScorerModule(LightningModule):
    def __init__(self,
                 embedder: SpanEmbedder,
                 span_scorer: SpanScorer,
                 pairwise_scorer: PairwiseScorer,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 criterion: torch.nn.Module,
                 bert_model
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.embedder = embedder
        self.span_scorer = span_scorer
        self.pairwise_scorer = pairwise_scorer
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loss = MeanMetric()
        self.evaluator = Metrics()
        self.labels = []

        self.bert_model = bert_model
        self.scores = []

    def forward(self, x):
        start_end, width, span_emb, span_labels, num_tokens = x
        span = self.embedder(start_end, span_emb, width)
        span_scores = self.span_scorer(span)
        pairwise_scores = self.pairwise_scorer(span_scores)
        return pairwise_scores

    def training_step(self, batch, batch_idx):
        spans = []
        labels = []
        first, second, labels = batch
        g1 = self.embedder(first)
        g2 = self.embedder(second)
        scores = self.pairwise_scorer(g1, g2)
        loss = self.criterion(scores, labels)
        self.log('train_loss',  loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.train_loss(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        first, second, labels = batch
        self.labels += labels.tolist()
        g1 = self.embedder(first)
        g2 = self.embedder(second)
        scores = self.pairwise_scorer(g1, g2)
        g1_scores = self.span_scorer(g1)
        g2_scores = self.span_scorer(g2)
        scores = g1_scores + g2_scores
        loss = self.criterion(scores, labels)
        eval = self.evaluator(scores, labels)
        self.log('val_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', eval['f1'], on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('val_precision', eval['precision'], on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('val_recall', eval['recall'], on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        first, second, labels = batch
        self.labels += labels.tolist()
        g1 = self.embedder(first)
        g2 = self.embedder(second)
        scores = self.pairwise_scorer(g1, g2)
        g1_scores = self.span_scorer(g1)
        g2_scores = self.span_scorer(g2)
        scores = g1_scores + g2_scores
        loss = self.criterion(scores, labels)
        eval = self.evaluator(scores, labels)
        self.log('test_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('test_f1', eval['f1'], on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('test_precision', eval['precision'], on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('test_recall', eval['recall'], on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

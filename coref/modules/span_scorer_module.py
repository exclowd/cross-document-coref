from typing import Dict
import torch
from lightning import LightningModule
from torchmetrics import MeanMetric

from transformers import AutoModel, AutoConfig
from coref.models import SpanScorer, SpanEmbedder


class SpanScorerModule(LightningModule):
    def __init__(self,
                 embedder: SpanEmbedder,
                 scorer: SpanScorer,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 criterion: torch.nn.Module,
                 bert_model
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.embedder = embedder
        self.scorer = scorer
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loss = MeanMetric()

        self.bert_model = bert_model

    def forward(self, x):
        start_end, width, span_emb, span_labels, num_tokens = x
        span = self.embedder(start_end, span_emb, width)
        span_scores = self.scorer(span)
        return span_scores

    def training_step(self, batch, batch_idx):
        start_end, width, span_emb, span_labels, num_tokens = batch
        span = self.embedder(start_end, span_emb, width)
        span_scores = self.scorer(span)
        loss = self.criterion(span_scores, span_labels)
        self.log('train_loss',  loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.train_loss(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = []
        total_tokens = 0
        start_end, width, span_emb, span_labels, num_tokens = batch
        labels += span_labels.tolist()
        total_tokens += num_tokens
        span = self.embedder(start_end, span_emb, width)
        span_scores = self.scorer(span)
        loss = self.criterion(span_scores, span_labels)
        self.log('val_loss', loss, on_step=True,
                    on_epoch=True, prog_bar=True, logger=True)
        return loss


    def test_step(self, batch, batch_idx):
        labels = []
        total_tokens = 0
        start_end, width, span_emb, span_labels, num_tokens = batch
        labels += span_labels.tolist()
        total_tokens += num_tokens
        span = self.embedder(start_end, span_emb, width)
        span_scores = self.scorer(span)
        loss = self.criterion(span_scores, span_labels)
        self.log('test_loss', loss, on_step=True,
                    on_epoch=True, prog_bar=True, logger=True)
        return loss
import hydra
import torch
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import Dictconf
from transformers import AutoTokenizer, AutoModel, AutoConfig

from coref.data.ecbplus import ECBPlusDataModule
from coref.modules.span_scorer_module import SpanScorerModule
import utils


def train(cfg: Dictconf):

    data_module: ECBPlusDataModule = instantiate(cfg.data)

    utils.fix_seed(cfg.random_seed)

    span_embedder = instantiate(cfg.span_embedder)

    bert_tokenizer = AutoTokenizer.from_pretrained(cfg.bert_tokenizer)

    bert_config = AutoConfig.from_pretrained(cfg.bert_tokenizer)

    bert_config.output_hidden_states = True

    bert_model = AutoModel.from_pretrained(
        cfg.bert_tokenizer, config=bert_config)

    span_scorer = instantiate(cfg.span_scorer)

    module = SpanScorerModule(
        embedder=span_embedder,
        scorer=span_scorer,
        optimizer=cfg.optimizer,
        scheduler=cfg.scheduler,
        criterion=cfg.criterion,
        bert_model=bert_model
    )

    trainer = Trainer(**cfg.trainer)

    trainer.fit(module, data_module=data_module)


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: Dictconf):
    train(cfg)

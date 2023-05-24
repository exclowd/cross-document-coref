import hydra
from omegaconf import DictConfig
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoModel, AutoConfig
import torch

from coref.data.ecbplus import ECBPlusDataModule


@hydra.main(version_base=None, config_path="../../conf", config_name="train")
def main(cfg: DictConfig):
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    clusterings = [AgglomerativeClustering(n_clusters=None,
                                           metric='precomputed',
                                           linkage=cfg['threshold']['linkage'],
                                           distance_threshold=threshold)
                   for threshold in thresholds]

    num_models = cfg['threshold']['num_models']

    # bert_cfg = AutoConfig.from_pretrained(cfg['bert_tokenizer'])
    # bert_cfg.output_hidden_states = True
    # bert_model = AutoModel.from_pretrained(cfg['bert_tokenizer'], config=bert_cfg)

    data = ECBPlusDataModule(cfg['data'], cfg['bert_tokenizer'])

    for model_num in range(num_models):

        # span_embedding
        embedder = hydra.utils.instantiate(cfg['span_embedder'])
        embedder.load_state_dict(torch.load(
            f"../../models/span_embedder_{model_num}.pt"))
        embedder.eval()

        # span_scorer
        scorer = hydra.utils.instantiate(cfg['span_scorer'])
        scorer.load_state_dict(torch.load(
            f"../../models/span_scorer_{model_num}.pt"))
        scorer.eval()

        # pairwise_scorer
        pairwise_scorer = hydra.utils.instantiate(cfg['pairwise_scorer'])
        pairwise_scorer.load_state_dict(torch.load(
            f"../../models/pairwise_scorer_{model_num}.pt"))
        pairwise_scorer.eval()

        for top_num, topic in enumerate(data.topic_list):
            print(f"Topic: {topic}")


if __name__ == '__main__':
    main()

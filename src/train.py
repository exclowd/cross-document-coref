import hydra
from omegaconf import Dictconf


def train():
    pass


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: Dictconf):
    train()

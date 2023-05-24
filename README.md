# Cross Document Coreference Resolution

Team Members:
Samyak Jain 2019101013
Kannav Mehta 2019101044

Paper that we implement: [here](https://aclanthology.org/2021.findings-acl.453.pdf)

### How to run

We first pre-train the span scorer and then train the pairwise scorer alongside the span scorer
We use hydra to manage all our configs and therefore the basic scripts will source their own configs.

```
python src/span_score_train.py
```

After we have done the pretraining then we train the pairwise scorer.

```
python src/train.py
```

We then continue to find the best model with the different thresholds and find scores

```
python src/eval.py
```

link to report: https://docs.google.com/document/d/1GLEbvwbL-V0bGU0E0SYFVUXYM3DQ-9yKVCaYZNtRxP0/edit#

link to slides: https://docs.google.com/presentation/d/1QT_nrlZIrPpkAbdMzM24W0UiJKo-ZHYzyjf3uu58a6w/edit#slide=id.g22c0f8fe7f3_0_72

link to models: 



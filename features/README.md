# Offline Feature Computation

This folder contains scripts for computing offline features used by the meta-network.

The offline feature computation consists of two parts:

1. Computing features from the generative model.
2. Computing features from the pretrained classifier without selection.

## 1. Features from Generative Model

First, run `extract_offline_feature.py` to extract image embeddings.
```bash
python extract_offline_feature.py
```

Then, run `calculate_feature.py` to calculate offline features from the generative model based on the extracted embeddings.
```bash
python calculate_feature.py
```

The generated features will be used as inputs to the selection network during training.

## 2. Features from Pretrained Classifier Without Selection

To compute features from the pretrained classifier without selection, run the provided shell script:

```bash
bash feature_scripts.sh
```

This script computes classifier-based offline features, such as gradient norms and forgetting-related features.

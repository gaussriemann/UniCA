# UniCA: Adapting Time Series Foundation Model to General Covariate-Aware Forecasting

Official implementation of **UniCA**, a unified framework for adapting Time Series Foundation Models (TSFMs) to
general covariate-aware forecasting tasks, including heterogeneous and multimodal inputs.

---

## 🧠 Overview

Time Series Foundation Models (TSFMs) achieve impressive generalization through large-scale pretraining, but struggle to
handle *heterogeneous covariates* (e.g., categorical features, images, text). **UniCA** addresses this limitation via:

* **Covariate Homogenization**: Transforms diverse covariates into a high-order homogeneous representation.
* **Attention-based Fusion**: Integrates covariate features while preserving TSFM’s temporal modeling capacity.
* **Plug-and-Play Adaptation**: UniCA acts as an adapter module that does **not modify** pretrained TSFM parameters.

UniCA supports a wide range of scenarios: single-modal, multimodal, homogeneous, and heterogeneous covariates.

---


## 📊 Datasets

We evaluate UniCA on both unimodal and multimodal datasets:

* 12 datasets, _e.g._ M5, Retail, EPF ... (unimodal time series)
* Time-MMD (text + time series)
* MMSP (image + time series)

---

## 🚀 Quick Start

### 1. Environment

create a new conda environment and install dependencies:

```bash
conda create -n unica python=3.10
conda activate unica
pip install -r requirements.txt
```

define the path to your pretrained model and dataset:

```bash
export MODEL_PATH=/your/pretrained/model/path
export DATA_PATH=/your/dataset/path
```

### 2. Data Preparation

Download the datasets from [drive](...) and place them in the `[DATA_PATH]` directory. The directory structure should
look like this:

```bash
[DATA_PATH]/
├── hog/
├── epf/
├── pdb/
└── ...
```

### 3. Reproduce results in paper

To reproduce the results in the paper, run the following command:

- For unimodal time series forecasting:

```bash
bash scripts/unimodal.sh
```

- For EPF subsets time series forecasting:

```bash
bash scripts/epf.sh
```

- For MMSP (image + time series) multimodal forecasting:

```bash
bash scripts/mmsp.sh
```

- For Time-MMD (text + time series) multimodal forecasting:

```bash
bash scripts/time-mmd.sh
```

---

## 🧪 Results

UniCA consistently improves performance over strong TSFM baselines across multiple metrics (MAE, MAPE, CRPS, etc.) and
covariate setups. See full results in our paper.



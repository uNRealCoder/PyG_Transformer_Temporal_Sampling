# Anti-Money Laundering Graph Neural Network (AML-GNN)

**Results at a glance:** https://uNRealCoder.github.io/PyG_Transformer_Temporal_Sampling/

Welcome to my AML-GNN project! This repository demonstrates how to use Graph Neural Networks (GNNs) and traditional machine learning to detect money laundering in financial transaction data. The project is designed for educational and portfolio purposes, showcasing practical data science and deep learning techniques.

## Project Overview

Money laundering is a major challenge for financial institutions. By modeling transactions as graphs, we can capture complex relationships between accounts and spot suspicious patterns more effectively. This project includes:
- **Data preprocessing**: Cleaning and transforming raw transaction data.
- **Baseline model**: Using XGBoost for initial classification.
- **Graph construction**: Building a graph from transaction data for GNN analysis.
- **GNN training and inference**: Training a custom GNN model and evaluating its performance.

## Dataset

https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

## Folder Structure

```
├── 1_PreProcess.ipynb        # Data cleaning and feature engineering
├── 2_Baseline.ipynb          # Baseline XGBoost model
├── 3_GraphCreate.ipynb       # Graph construction and GNN training
├── 4_GraphInference.ipynb    # GNN inference and evaluation
├── GNN_model.py              # Custom GNN model definition
├── custom_earlystop.py       # Early stopping utility for training
├── processed_data/           # Preprocessed data and mappings
├── archive/                  # Raw data files: You need to download this from above URL.
```

## How It Works

1. **Preprocessing**: The first notebook loads raw CSVs, encodes categorical features, and splits the data into train/test sets.
2. **Baseline Model**: The second notebook trains an XGBoost classifier to provide a reference performance.
3. **Graph Construction**: The third notebook builds a graph from transactions, assigns node/edge features, and trains a GNN using PyTorch Geometric.
4. **Inference**: The final notebook loads the trained GNN and evaluates it on the test set, visualizing results with confusion matrices.

## Key Technologies
- Python (Pandas, NumPy, Scikit-learn)
- PyTorch & PyTorch Geometric
- XGBoost
- Jupyter Notebooks

## Getting Started

1. **Clone the repo**:
   ```bash
   git clone https://github.com/uNRealCoder/PyG_Transformer_Temporal_Sampling.git
   ```

2. **Run the notebooks** in order (1 → 4) to preprocess data, train models, and evaluate results.

## Results

- The project compares traditional ML and GNN approaches for AML detection.
- Confusion matrices and feature importance plots help visualize model performance. You can see that XGBoost has been beaten with Graph Transformers
- The GNN model leverages graph structure for improved detection of laundering activities.

---

**Author:** Nikhil Ranjan

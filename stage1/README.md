# Graph-Based Time Series Data Classification

A three-stage graph-based pipeline for constructing functional brain networks from fMRI time-series data, building a subject-similarity graph, and classifying subjects into Autism and Control groups using a Graph Convolutional Network (GCN).

---

## Overview

This project implements a three-stage graph-based pipeline for analyzing fMRI time-series data.

The pipeline consists of:

1. **Stage 1:** Construction of functional brain networks from fMRI time-series data.
2. **Stage 2:** Construction of a subject-similarity graph based on the functional connectivity patterns of subjects.
3. **Stage 3:** Classification of subjects into Autism and Control groups using a Graph Convolutional Network (GCN).

The project was developed and tested using a specific fMRI dataset. The same dataset should be used to reproduce the project results.

---

## Project Objectives

The main objectives of this project are:

* Construct functional brain networks from fMRI time-series data.
* Analyze connectivity between brain regions.
* Represent relationships between subjects using a similarity graph.
* Extract graph-based features from subjects.
* Apply a Graph Convolutional Network for subject classification.
* Evaluate classification performance using accuracy and a confusion matrix.

---

# Dataset

This project uses fMRI time-series data and phenotypic information from the **Autism Brain Imaging Data Exchange (ABIDE)** dataset.

The input dataset used in this project contains:

* **1035 `.1D` fMRI files**
* `Phenotypic_V1_0b_preprocessed1.csv`

Each `.1D` file contains fMRI time-series data where:

* Rows represent time points.
* Columns represent brain regions of interest (ROIs).

The phenotypic CSV file contains subject information including:

* `SUB_ID`
* `DX_GROUP`

For this project:

* `DX_GROUP = 1` → Autism
* `DX_GROUP = 2` → Control

---

## Download the Dataset

The exact input dataset used in this project is available through the following Google Drive link:

**[Download the Project Dataset](https://drive.google.com/file/d/19MvA3VDgXzv9KyyYDljTH1xss3vcmSyz/view?usp=sharing)**

Download the dataset before running the project.

---

## Dataset Placement

The Python scripts are configured to look for the input data on the user's Desktop.

After downloading and extracting the dataset, place the files on your Desktop using the following structure:

```text
Desktop/
│
├── rois_aal/
│   ├── *.1D
│   ├── *.1D
│   ├── ...
│   └── *.1D
│
└── Phenotypic_V1_0b_preprocessed1.csv
```

The folder containing the `.1D` files must be named exactly:

```text
rois_aal
```

The phenotypic file must be placed directly on the Desktop and must have the exact filename:

```text
Phenotypic_V1_0b_preprocessed1.csv
```

### Important

The dataset used in this project is the specific dataset used during development and testing.

Using a different dataset may produce different results or may require modifications to the code.

---

# Stage 1: Functional Brain Network Construction

Stage 1 processes the fMRI time-series data of individual subjects and constructs functional brain networks.

For each subject:

1. The fMRI time-series matrix is loaded.
2. A Pearson correlation matrix is calculated between ROIs.
3. Several correlation thresholds are evaluated.
4. A threshold is selected based on the resulting graph density.
5. A functional brain network is constructed.
6. Edges represent functional connectivity between brain regions.
7. Graph information and summary reports are saved.

### Correlation Matrix

The Pearson correlation coefficient is used to measure the relationship between pairs of ROIs.

The correlation matrix is calculated using:

```python
np.corrcoef(matrix.T)
```

### Thresholding

Candidate thresholds are evaluated:

```text
0.3
0.4
0.5
0.6
0.7
```

The selected threshold is based on the resulting number of graph edges.

### Stage 1 Outputs

The generated results are saved on the Desktop:

```text
Desktop/stage1_outputs/
```

Example outputs include:

```text
stage1_outputs/
├── threshold_graph_edges.csv
├── person_report.csv
└── ...
```

---

# Stage 2: Subject Similarity Graph Construction

Stage 2 constructs a graph representing the similarity between subjects.

For each subject:

1. The functional connectivity matrix is calculated.
2. The upper triangular part of the correlation matrix is extracted.
3. This upper triangular part is used as the subject feature vector.
4. Cosine similarity is calculated between subjects.
5. A subject-similarity graph is constructed.
6. Subjects are represented as nodes.
7. Similarity relationships are represented as edges.

### Cosine Similarity

Cosine similarity is used to measure the similarity between the feature vectors of subjects.

Candidate similarity thresholds are evaluated:

```text
0.10
0.20
0.30
0.40
0.50
0.60
```

### Stage 2 Outputs

The generated results are saved on the Desktop:

```text
Desktop/stage2_outputs/
```

The main output used by Stage 3 is:

```text
person_graph_edges.csv
```

---

# Stage 3: Graph-Based Classification with GCN

Stage 3 uses a **Graph Convolutional Network (GCN)** to classify subjects into Autism and Control groups.

The subject-similarity graph generated in Stage 2 is used as the input graph.

## Node Features

Two graph-based features are used for each subject:

* Degree
* Clustering coefficient

Therefore, each subject is represented using two node features.

## GCN Architecture

The model contains two neural network layers:

```text
Input Features
      ↓
Graph Convolution Layer
      ↓
ReLU Activation
      ↓
Graph Convolution Layer
      ↓
Output Classes
```

The model is implemented using PyTorch.

## Training Configuration

The current implementation uses:

* Hidden dimension: `16`
* Output classes: `2`
* Learning rate: `0.01`
* Optimizer: Adam
* Training epochs: `200`
* Loss function: Cross-Entropy Loss
* Train/Test split: `80/20`
* Random state: `42`

The labels are derived from the `DX_GROUP` column:

```text
DX_GROUP = 1 → Autism → class 0
DX_GROUP = 2 → Control → class 1
```

## Stage 3 Outputs

The generated results are saved on the Desktop:

```text
Desktop/stage3_outputs/
```

Important output files include:

```text
stage3_report.csv
prediction_results.csv
```

---

# Project Structure

The GitHub repository contains the source code and project documentation:

```text
AD-Graph-Based-Classification/
│
├── stage1/
│   └── stage1.py
│
├── stage2/
│   └── stage2.py
│
├── stage3/
│   └── stage3.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

The input dataset is downloaded separately from the Google Drive link provided above and placed on the Desktop before running the project.

---

# Installation

## Windows

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## macOS / Linux

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Before running the code, make sure the dataset has been downloaded and placed correctly on the Desktop.

The three stages should be executed in order.

## Step 1 — Run Stage 1

```bash
python stage1/stage1.py
```

Stage 1 reads the fMRI input files from:

```text
Desktop/rois_aal/
```

and generates:

```text
Desktop/stage1_outputs/
```

---

## Step 2 — Run Stage 2

After Stage 1 is completed, run:

```bash
python stage2/stage2.py
```

Stage 2 generates:

```text
Desktop/stage2_outputs/
```

including:

```text
person_graph_edges.csv
```

---

## Step 3 — Run Stage 3

After Stage 2 is completed, run:

```bash
python stage3/stage3.py
```

Stage 3 reads the subject graph and phenotypic information and generates:

```text
Desktop/stage3_outputs/
```

including:

```text
stage3_report.csv
prediction_results.csv
```

---

# Reproducibility

Random seeds are used in different parts of the project to improve reproducibility.

For example:

```python
random.seed(42)
```

and:

```python
random_state=42
```

The same input dataset, code, and configuration should be used when attempting to reproduce the results.

---

# Design Decisions

## Stage 1

A threshold-based functional connectivity graph is constructed from the ROI correlation matrix.

Candidate thresholds are evaluated, and a threshold is selected according to the resulting number of graph edges.

Each graph node represents an ROI, while edges represent functional relationships between ROIs.

## Stage 2

The upper triangular part of each subject's correlation matrix is used as the subject feature vector.

Cosine similarity is then used to measure similarity between subjects.

The resulting graph represents subjects as nodes and their similarity relationships as edges.

## Stage 3

A Graph Convolutional Network is used for graph-based classification.

The current implementation uses:

* Degree
* Clustering coefficient

as node features.

The adjacency matrix used by the GCN is binary and includes self-loops with symmetric normalization.

---

# Implementation Notes and Limitations

* The project was developed and tested using the specified input dataset.
* Results may change when a different dataset is used.
* Stage 1 and Stage 2 use threshold selection based on graph edge counts.
* The current GCN implementation uses binary adjacency rather than the original similarity values as adjacency weights.
* Only degree and clustering coefficient are currently used as node features.
* The project is an academic implementation and is not intended for clinical diagnosis.
* The generated output directories are created on the user's Desktop and are not required to be stored in the GitHub repository.

---

# References

* Di Martino, A., et al. (2014). The Autism Brain Imaging Data Exchange: Towards a Large-Scale Evaluation of the Intrinsic Brain Architecture in Autism. *Molecular Psychiatry*.
* Bullmore, E., & Sporns, O. (2009). Complex brain networks: graph theoretical analysis of structural and functional systems. *Nature Reviews Neuroscience*.
* Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks.
* PyTorch Documentation.
* Scikit-learn Documentation.
* NetworkX Documentation.

---

# Academic Context

This project was developed as an academic project involving:

* fMRI time-series analysis
* Functional connectivity
* Graph-based data representation
* Subject similarity modeling
* Graph neural networks
* Graph Convolutional Networks
* Autism vs. Control classification

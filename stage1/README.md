# Graph-Based Time Series Data Classification

A three-stage graph-based pipeline for constructing functional brain networks from fMRI time-series data, building a subject-similarity graph, and classifying subjects into Autism and Control groups using a Graph Convolutional Network (GCN).

## Overview

This project implements a three-stage graph-based pipeline for analyzing fMRI time-series data.

The pipeline consists of:

1. **Stage 1:** Construction of functional brain networks from fMRI time-series data.
2. **Stage 2:** Construction of a subject-similarity graph based on the functional connectivity patterns of subjects.
3. **Stage 3:** Classification of subjects into Autism and Control groups using a Graph Convolutional Network (GCN).

The project was developed and tested using a specific fMRI dataset. Therefore, the same input dataset should be used to reproduce the reported results.

---

## Project Objectives

The main objectives of this project are:

* Construct functional brain networks from fMRI time-series data.
* Analyze the connectivity between brain regions.
* Represent relationships between subjects using a similarity graph.
* Extract graph-based features from subjects.
* Apply a Graph Convolutional Network for subject classification.
* Evaluate the classification performance using accuracy and a confusion matrix.

---

## Dataset

This project uses fMRI time-series data and phenotypic information from the **Autism Brain Imaging Data Exchange (ABIDE)** dataset.

The dataset used in this project contains:

* **1035 `.1D` fMRI files**
* A phenotypic CSV file:
  `Phenotypic_V1_0b_preprocessed1.csv`

Each `.1D` file contains fMRI time-series data, where:

* Rows represent time points.
* Columns represent brain regions of interest (ROIs).

The phenotypic file contains subject information including:

* `SUB_ID`
* `DX_GROUP`

For this project:

* `DX_GROUP = 1` → Autism
* `DX_GROUP = 2` → Control

### Where to Get the Dataset

Download the dataset used in this project from the following source:

**Dataset source:**
[INSERT DATASET DOWNLOAD LINK HERE]

The project should be run using the same dataset described above in order to reproduce the results.

### Dataset Placement

After downloading and extracting the dataset, place the input files on your **Desktop** using the following structure:

```text
Desktop/
│
├── rois_aal/
│   ├── subject_1.1D
│   ├── subject_2.1D
│   ├── ...
│   └── subject_1035.1D
│
└── Phenotypic_V1_0b_preprocessed1.csv
```

The folder containing the `.1D` files must be named:

```text
rois_aal
```

and the phenotypic file must be placed directly on the Desktop with the exact filename:

```text
Phenotypic_V1_0b_preprocessed1.csv
```

The code automatically uses the current user's Desktop directory, so the absolute Windows username does not need to be changed in the source code.

### Important

The input dataset is required before running the project.

The three stages are designed to be executed sequentially:

```text
Input Dataset
      ↓
Stage 1
      ↓
Stage 2
      ↓
Stage 3
```

Using a different dataset may result in different outputs or may require modifications to the code.

---

## Stage 1: Functional Brain Network Construction

Stage 1 processes the fMRI time-series data for individual subjects.

For each subject:

1. The fMRI time-series matrix is loaded.
2. A Pearson correlation matrix is calculated between ROIs.
3. Different correlation thresholds are evaluated.
4. A threshold is selected based on the resulting graph density.
5. A functional brain network is constructed.
6. Edge weights represent correlations between brain regions.
7. Graph information and summary reports are saved.

The generated files are stored in:

```text
Desktop/stage1_outputs/
```

---

## Stage 2: Subject Similarity Graph Construction

Stage 2 represents relationships between subjects.

For each subject:

1. The functional connectivity matrix is calculated.
2. The upper triangular part of the correlation matrix is extracted as a feature vector.
3. Cosine similarity is calculated between subjects.
4. A subject-similarity graph is constructed.
5. Subjects are represented as graph nodes.
6. Similarity relationships are represented as graph edges.

The generated files are stored in:

```text
Desktop/stage2_outputs/
```

The main output used by Stage 3 is:

```text
person_graph_edges.csv
```

---

## Stage 3: Graph-Based Classification with GCN

Stage 3 uses a Graph Convolutional Network (GCN) to classify subjects into Autism and Control groups.

The graph constructed in Stage 2 is used as the input graph.

Node features include:

* Degree
* Clustering coefficient

The GCN consists of two graph convolutional layers with a ReLU activation function between them.

The model is trained using:

* Adam optimizer
* Learning rate: `0.01`
* 200 training epochs
* Cross-entropy loss

The dataset is divided into training and test sets using an 80/20 split with stratification.

The generated results are stored in:

```text
Desktop/stage3_outputs/
```

Important output files include:

```text
stage3_report.csv
prediction_results.csv
```

---

## Project Structure

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

The input dataset is placed on the user's Desktop as described in the **Dataset Placement** section.

---

## Installation

### Windows

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### macOS / Linux

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Make sure the input dataset has been placed correctly on the Desktop before running the code.

Run the stages in order:

### Stage 1

```bash
python stage1/stage1.py
```

### Stage 2

```bash
python stage2/stage2.py
```

### Stage 3

```bash
python stage3/stage3.py
```

Each stage uses the output generated by the previous stage.

---

## Reproducibility

Random seeds are used in different parts of the project to improve reproducibility.

For example:

```python
random.seed(42)
```

and:

```python
random_state=42
```

The same input dataset and configuration should be used when attempting to reproduce the results.

---

## Design Decisions

### Stage 1

A threshold-based functional connectivity graph is constructed from the ROI correlation matrix.

Candidate thresholds are evaluated and a threshold is selected based on the resulting number of graph edges.

### Stage 2

The upper triangular part of each subject's correlation matrix is used as the subject feature vector.

Cosine similarity is then used to measure similarity between subjects.

### Stage 3

A Graph Convolutional Network is used to perform graph-based classification.

The current implementation uses two node features:

* Degree
* Clustering coefficient

The adjacency matrix used by the GCN is binary and includes self-loops with symmetric normalization.

---

## Implementation Notes and Limitations

* The project was developed and tested using the specified input dataset.
* Results may change when using a different dataset.
* Stage 1 and Stage 2 use threshold selection based on graph edge counts.
* The current GCN implementation uses binary adjacency rather than the original similarity values as edge weights.
* Only degree and clustering coefficient are currently used as node features.
* The project is intended as an academic implementation and is not a clinical diagnostic system.

---

## References

* Di Martino, A., et al. (2014). The Autism Brain Imaging Data Exchange: Towards a Large-Scale Evaluation of the Intrinsic Brain Architecture in Autism. *Molecular Psychiatry*.
* Bullmore, E., & Sporns, O. (2009). Complex brain networks: graph theoretical analysis of structural and functional systems. *Nature Reviews Neuroscience*.
* Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks.
* PyTorch Documentation.
* Scikit-learn Documentation.
* NetworkX Documentation.

---

## Academic Context

This project was developed as an academic project involving graph-based data analysis, functional connectivity, subject similarity modeling, and graph neural networks.

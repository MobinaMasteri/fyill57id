Graph-Based Time Series Data Classification

Overview

This project presents a three-stage graph-based pipeline for analyzing fMRI time-series data and performing graph-based classification.

The project converts fMRI time-series data into functional brain networks, constructs a graph representing similarities between subjects, and finally uses a Graph Convolutional Network (GCN) to classify subjects into Autism and Control groups.

The project consists of three main stages:

1. Stage 1: Construction and analysis of functional brain networks.
2. Stage 2: Construction of a subject-similarity graph.
3. Stage 3: Classification using a Graph Convolutional Network (GCN).

---

Objectives

The main objectives of this project are:

- Construct functional brain networks from fMRI time-series data.
- Calculate functional connectivity between brain regions.
- Analyze the connectivity structure of individual subjects.
- Calculate similarity between subjects.
- Construct a graph representing relationships between subjects.
- Extract graph-based features from the subject graph.
- Train a Graph Convolutional Network (GCN).
- Classify subjects into Autism and Control groups.
- Evaluate the classification performance using accuracy and a confusion matrix.

---

Dataset

This project uses fMRI data from the ABIDE (Autism Brain Imaging Data Exchange) dataset.

The dataset used by this project contains:

- 1035 ".1D" fMRI time-series files
- "Phenotypic_V1_0b_preprocessed1.csv"

Dataset Files

1. "rois_aal"

The "rois_aal" folder contains the fMRI time-series files.

Each ".1D" file contains time-series data for multiple brain regions (ROIs).

The rows represent time points, while the columns represent brain regions.

2. "Phenotypic_V1_0b_preprocessed1.csv"

This CSV file contains the phenotypic information used in Stage 3.

The main columns used by the project are:

- "SUB_ID" — Subject ID
- "DX_GROUP" — Diagnostic group

The labels are interpreted as:

DX_GROUP = 1 → Autism
DX_GROUP = 2 → Control

---

Dataset Download

The dataset is not included in this GitHub repository because of its size.

The dataset should be downloaded separately using the provided Google Drive link.

After downloading the dataset, the files should be placed inside a local "data" directory in the project.

The expected structure is:

AD-Graph-Based-Classification/
│
├── data/
│   ├── rois_aal/
│   │   ├── *.1D
│   │   ├── ...
│   │
│   └── Phenotypic_V1_0b_preprocessed1.csv
│
├── stage1/
├── stage2/
├── stage3/
├── README.md
├── requirements.txt
└── .gitignore

Important

The "data" folder is used locally and the dataset itself does not need to be uploaded to GitHub.

After cloning the repository, create the "data" folder and place the downloaded dataset inside it.

The folder and file names should remain exactly:

rois_aal
Phenotypic_V1_0b_preprocessed1.csv

The project code expects these exact names.

«Note: Using a different dataset, changing the dataset structure, or changing the file names may produce different results or require modifications to the code.»

---

Project Structure

The GitHub repository contains the source code and project documentation, but does not contain the dataset.

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

The dataset is downloaded separately and placed locally inside:

data/
├── rois_aal/
└── Phenotypic_V1_0b_preprocessed1.csv

---

Stage 1 — Functional Brain Network Construction

Stage 1 constructs a functional brain network for each subject.

Step 1: Reading fMRI Data

Each ".1D" file is loaded as a numerical matrix.

The rows represent time points and the columns represent brain regions.

---

Step 2: Correlation Matrix

Pearson correlation is calculated between the time-series of all brain regions.

The correlation matrix is calculated using:

np.corrcoef(matrix.T)

Each element of the resulting matrix represents the functional correlation between two ROIs.

---

Step 3: Threshold Selection

Several candidate thresholds are tested:

0.3
0.4
0.5
0.6
0.7

The program randomly selects 100 subjects using:

random.seed(42)

For each threshold, the average number of edges in the resulting graphs is calculated.

The first threshold producing an average number of edges between:

100 ≤ edges ≤ 200

is selected.

If no threshold satisfies this condition, the fallback threshold is:

0.5

---

Step 4: Threshold Graph

An edge is created between two ROIs when:

abs(correlation) >= threshold

Each edge contains:

- Correlation
- Distance

The distance is calculated as:

1 - abs(correlation)

---

Step 5: Strongest and Weakest Connections

For each subject, the strongest and weakest ROI connections are identified based on the absolute correlation value.

---

Stage 1 Outputs

The results are saved in:

stage1_outputs/

For each subject, the program creates a separate directory containing:

threshold_graph_edges.csv
person_report.csv

For up to 10 subjects, a graph visualization is also generated:

threshold_graph.png

---

Stage 2 — Subject Similarity Graph

Stage 2 constructs a graph where:

- Each node represents one subject.
- Each edge represents similarity between two subjects.

Step 1: Correlation Feature Vector

For every subject, the Pearson correlation matrix is calculated.

Only the upper triangular part of the correlation matrix is extracted.

This produces one feature vector for each subject.

---

Step 2: Cosine Similarity

Cosine similarity is calculated between the feature vectors of every pair of subjects.

The similarity is calculated using:

cosine_similarity()

---

Step 3: Similarity Threshold

The following candidate thresholds are tested:

0.10
0.20
0.30
0.40
0.50
0.60

100 subjects are randomly selected using:

random.seed(42)

The first threshold producing between:

100 ≤ edges ≤ 1500

is selected.

If no threshold satisfies this condition, the fallback threshold is:

0.3

---

Step 4: Subject Graph

Each subject becomes a node.

An edge is created when:

similarity >= similarity_threshold

The similarity value is stored as an edge attribute.

---

Stage 2 Outputs

The results are saved in:

stage2_outputs/

The main files are:

person_graph_edges.csv
graph_report.csv

The program also generates up to 10 sample graph visualizations.

---

Stage 3 — Graph Convolutional Network Classification

Stage 3 uses the subject-similarity graph created in Stage 2.

The goal is to classify subjects into:

Autism
Control

---

Step 1: Loading the Subject Graph

The following file from Stage 2 is used:

stage2_outputs/person_graph_edges.csv

The subject graph is reconstructed using NetworkX.

---

Step 2: Loading Labels

The phenotypic file:

Phenotypic_V1_0b_preprocessed1.csv

is used to obtain the diagnostic labels.

The mapping is:

DX_GROUP = 1 → Autism
DX_GROUP = 2 → Control

---

Step 3: Node Features

Two graph-based features are calculated for each subject:

1. Degree
2. Clustering coefficient

Therefore, every subject is represented using two node features:

[degree, clustering]

---

Step 4: Graph Adjacency Matrix

A binary adjacency matrix is constructed from the subject graph.

Self-loops are added to the adjacency matrix.

The adjacency matrix is then symmetrically normalized using the node degrees.

---

Step 5: GCN Architecture

The model consists of two graph convolutional layers implemented using PyTorch linear layers.

The architecture is:

Input
  ↓
Linear Layer
  ↓
ReLU
  ↓
Linear Layer
  ↓
Output

The configuration is:

Input dimension: 2
Hidden dimension: 16
Output dimension: 2

The model uses:

Optimizer: Adam
Learning rate: 0.01
Epochs: 200
Loss function: Cross Entropy

---

Step 6: Train/Test Split

The dataset is divided into:

80% → Training
20% → Testing

The split uses:

random_state=42

and stratification is used to preserve the class distribution.

---

Stage 3 Outputs

The results are saved in:

stage3_outputs/

The generated files are:

stage3_report.csv
prediction_results.csv

The report contains:

- Number of subjects
- Number of Autism subjects
- Number of Control subjects
- Accuracy
- Confusion matrix values

---

Installation

Make sure Python is installed on your system.

Clone the repository and enter the project directory.

Then install the required packages:

pip install -r requirements.txt

The required libraries include:

numpy
pandas
networkx
matplotlib
scikit-learn
torch

---

Dataset Setup

After downloading the dataset:

1. Open the project folder.
2. Create a folder named:

data

3. Put the "rois_aal" folder inside "data".
4. Put "Phenotypic_V1_0b_preprocessed1.csv" inside "data".

The final structure should be:

AD-Graph-Based-Classification/
│
├── data/
│   ├── rois_aal/
│   │   ├── *.1D
│   │   └── ...
│   │
│   └── Phenotypic_V1_0b_preprocessed1.csv
│
├── stage1/
├── stage2/
├── stage3/
├── README.md
├── requirements.txt
└── .gitignore

---

Running the Project

The three stages should be executed in order.

Stage 1

python stage1/stage1.py

Stage 1 creates:

stage1_outputs/

---

Stage 2

After Stage 1 is completed:

python stage2/stage2.py

Stage 2 creates:

stage2_outputs/

The main file used by Stage 3 is:

stage2_outputs/person_graph_edges.csv

---

Stage 3

After Stage 2 is completed:

python stage3/stage3.py

Stage 3 creates:

stage3_outputs/

---

Reproducibility

The project uses fixed random seeds in the relevant parts of the pipeline.

For example:

random.seed(42)

and:

random_state=42

This helps make the sampling and train/test split reproducible.

---

Design Decisions

Several design decisions were made in this implementation.

Stage 1

The functional connectivity matrix is constructed using Pearson correlation.

The graph is created by applying an absolute correlation threshold.

Stage 2

The complete upper-triangular correlation matrix is used as the feature vector for each subject.

Cosine similarity is then used to measure similarity between subjects.

Stage 3

The subject graph is represented using binary adjacency.

Although Stage 2 stores similarity values on its edges, the Stage 3 adjacency matrix uses binary connections rather than similarity values as edge weights.

The node features currently consist of:

Degree
Clustering coefficient

---

Limitations

The current implementation has several limitations:

- The dataset is not included in the GitHub repository.
- Results depend on the selected dataset and preprocessing.
- Threshold selection is based on graph edge counts.
- Stage 3 uses only two node features.
- Stage 3 uses binary adjacency rather than weighted similarity values.
- The model uses a single train/test split.
- No cross-validation is performed.
- The current implementation is intended as an academic/project implementation rather than a production system.

---

References

- Di Martino, A. et al. (2014). The Autism Brain Imaging Data Exchange: Towards a Large-Scale Evaluation of the Intrinsic Brain Architecture in Autism.
- Bullmore, E., & Sporns, O. (2009). Complex brain networks: graph theoretical analysis of structural and functional systems.
- Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks.
- PyTorch Documentation.
- scikit-learn Documentation.
- NetworkX Documentation.

---

Academic Context

This project was developed as an academic implementation of graph-based analysis and classification of fMRI time-series data.

The complete pipeline demonstrates the following concepts:

fMRI Time-Series
       ↓
Correlation Matrix
       ↓
Functional Brain Graph
       ↓
Subject Feature Vectors
       ↓
Subject Similarity Graph
       ↓
Graph Features
       ↓
Graph Convolutional Network
       ↓
Autism / Control Classification
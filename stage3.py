import re
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"

STAGE2_DIR = PROJECT_DIR / "stage2_outputs"

PHENOTYPIC_FILE = DATA_DIR / "Phenotypic_V1_0b_preprocessed1.csv"

OUTPUT_DIR = PROJECT_DIR / "stage3_outputs"

def read_person_graph(edge_file):

    graph = nx.Graph()

    data = pd.read_csv(edge_file)

    for _, row in data.iterrows():

        graph.add_edge(
            row["person_1"],
            row["person_2"],
            similarity=row["similarity"]
        )

    return graph


def extract_subject_id(person_name):

    numbers = re.findall(r"\d+", person_name)

    if len(numbers) == 0:
        return None

    return int(numbers[-1])


def load_labels(phenotypic_file):

    data = pd.read_csv(phenotypic_file)

    labels = {}

    for _, row in data.iterrows():

        subject_id = int(row["SUB_ID"])

        dx_group = int(row["DX_GROUP"])

        labels[subject_id] = dx_group

    return labels


class GCN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):

        super().__init__()

        self.layer1 = nn.Linear(input_dim, hidden_dim)

        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adjacency):

        x = torch.matmul(adjacency, x)

        x = self.layer1(x)

        x = F.relu(x)

        x = torch.matmul(adjacency, x)

        x = self.layer2(x)

        return x


def build_gnn_dataset(graph, labels):

    valid_nodes = []

    x = []

    y = []

    total_nodes = graph.number_of_nodes()

    for index, node in enumerate(graph.nodes(), start=1):

        print(f"[{index}/{total_nodes}] Processing {node}")

        subject_id = extract_subject_id(node)

        if subject_id not in labels:
            continue

        degree = graph.degree(node)

        clustering = nx.clustering(graph, node)

        x.append([
            degree,
            clustering
        ])

        y.append(labels[subject_id] - 1)

        valid_nodes.append(node)

    node_to_index = {}

    for index, node in enumerate(valid_nodes):

        node_to_index[node] = index

    adjacency = torch.zeros(
        (len(valid_nodes), len(valid_nodes)),
        dtype=torch.float32
    )

    for person_1, person_2 in graph.edges():

        if person_1 in node_to_index and person_2 in node_to_index:

            i = node_to_index[person_1]

            j = node_to_index[person_2]

            adjacency[i][j] = 1

            adjacency[j][i] = 1

    adjacency = adjacency + torch.eye(len(valid_nodes))

    degree = adjacency.sum(dim=1)

    degree_inv_sqrt = torch.pow(degree, -0.5)

    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0

    D = torch.diag(degree_inv_sqrt)

    adjacency = D @ adjacency @ D

    x = torch.tensor(x, dtype=torch.float32)

    y = torch.tensor(y, dtype=torch.long)

    return x, y, adjacency, valid_nodes


def train_gnn(x, y, adjacency, person_names):

    indices = np.arange(len(y))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=y.numpy()
    )

    model = GCN(
        input_dim=x.shape[1],
        hidden_dim=16,
        output_dim=2
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01
    )

    for epoch in range(200):

        model.train()

        output = model(x, adjacency)

        loss = F.cross_entropy(
            output[train_indices],
            y[train_indices]
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if epoch % 20 == 0:

            print(
                f"Epoch {epoch}, Loss: {loss.item():.4f}"
            )

    model.eval()

    with torch.no_grad():

        output = model(x, adjacency)

        predictions = output.argmax(dim=1)

    y_test = y[test_indices].numpy()

    y_pred = predictions[test_indices].numpy()

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    conf_matrix = confusion_matrix(
        y_test,
        y_pred
    )

    results = pd.DataFrame({
        "person_name": [person_names[i] for i in test_indices],
        "real_label": y_test + 1,
        "predicted_label": y_pred + 1
    })

    return model, accuracy, conf_matrix, results


def save_stage3_report(output_path, y, accuracy, conf_matrix):

    autism_count = torch.sum(y == 0).item()

    control_count = torch.sum(y == 1).item()

    with open(output_path, "w", encoding="utf-8") as f:

        f.write("item,value\n")

        f.write(f"number_of_people,{len(y)}\n")

        f.write(f"autism_count,{autism_count}\n")

        f.write(f"control_count,{control_count}\n")

        f.write(f"accuracy,{accuracy:.6f}\n")

        f.write(f"true_autism,{conf_matrix[0][0]}\n")

        f.write(f"false_control,{conf_matrix[0][1]}\n")

        f.write(f"false_autism,{conf_matrix[1][0]}\n")

        f.write(f"true_control,{conf_matrix[1][1]}\n")


def save_prediction_results(
    results,
    output_path
):

    results.to_csv(
        output_path,
        index=False
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Reading person graph...")

    graph = read_person_graph(
        STAGE2_DIR / "person_graph_edges.csv"
    )

    print("Loading labels...")

    labels = load_labels(
        PHENOTYPIC_FILE
    )

    print("Building GNN dataset...")

    x, y, adjacency, person_names = build_gnn_dataset(
        graph,
        labels
    )

    print("Training GCN model...")

    model, accuracy, conf_matrix, results = train_gnn(
        x,
        y,
        adjacency,
        person_names
    )

    print("Saving reports...")

    save_stage3_report(
        OUTPUT_DIR / "stage3_report.csv",
        y,
        accuracy,
        conf_matrix
    )

    save_prediction_results(
        results,
        OUTPUT_DIR / "prediction_results.csv"
    )

    print(f"Accuracy: {accuracy:.4f}")

    print("Stage 3 completed.")

    print("Outputs saved in:", OUTPUT_DIR)


if __name__ == "__main__":

    main()

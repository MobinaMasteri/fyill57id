import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"

ROI_DIR = DATA_DIR / "rois_aal"
OUTPUT_DIR = PROJECT_DIR / "stage2_outputs"

def read_fmri_file(file_path):

    matrix = np.loadtxt(file_path)

    return matrix

def build_correlation_matrix(matrix):

    corr_matrix = np.corrcoef(matrix.T)

    corr_matrix = np.nan_to_num(corr_matrix)

    return corr_matrix

def extract_person_feature_vector(file_path):

    matrix = read_fmri_file(file_path)

    corr_matrix = build_correlation_matrix(matrix)

    upper_triangle = np.triu_indices_from(corr_matrix, k=1)

    feature_vector = corr_matrix[upper_triangle]

    return feature_vector

def build_person_features(roi_files):

    person_features = {}

    for file_path in roi_files:

        person_name = file_path.stem

        feature_vector = (extract_person_feature_vector(file_path))

        person_features[person_name] = feature_vector

    return person_features

def calculate_similarity(vector_1, vector_2):

    similarity = cosine_similarity([vector_1], [vector_2]).item()

    return similarity

def build_person_graph(person_features, similarity_threshold):

    graph = nx.Graph()

    person_names = list(person_features.keys())

    for person_name in person_names:
        graph.add_node(person_name)

    total_people = len(person_names)

    for i in range(total_people):

        print(f"[{i + 1}/{total_people}] Processing {person_names[i]}")

        for j in range(i + 1, total_people):

            person_1 = person_names[i]
            person_2 = person_names[j]

            similarity = calculate_similarity(person_features[person_1], person_features[person_2])

            if similarity >= similarity_threshold:

                graph.add_edge(person_1, person_2, similarity=similarity)

    return graph

def save_person_graph_edges(graph, output_path):

    with open(output_path, "w", encoding="utf-8") as f:

        f.write("person_1,person_2,similarity\n")

        for person_1, person_2, edge_data in graph.edges(data=True):

            f.write(f"{person_1},"
                f"{person_2},"
                f"{edge_data['similarity']:.6f}\n")
            
def save_graph_report(graph, similarity_threshold, output_path):

    with open(output_path, "w", encoding="utf-8") as f:

        f.write("item,value\n")

        f.write( f"similarity_threshold,"
            f"{similarity_threshold}\n")

        f.write(f"number_of_people,"
            f"{graph.number_of_nodes()}\n")

        f.write(f"number_of_edges,"
            f"{graph.number_of_edges()}\n")

        f.write(f"density,"
            f"{nx.density(graph):.6f}\n")
        
SIMILARITY_VALUES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
RANDOM_DATA = 100
def choose_similarity_threshold(person_features):

    random.seed(42)

    person_names = list(person_features.keys())

    random_people = random.sample(person_names, RANDOM_DATA)

    for threshold in SIMILARITY_VALUES:

        graph = nx.Graph()

        for person in random_people:
            graph.add_node(person)

        for i in range(len(random_people)):

            for j in range(i + 1, len(random_people)):

                person_1 = random_people[i]
                person_2 = random_people[j]

                similarity = calculate_similarity(person_features[person_1], person_features[person_2])

                if similarity >= threshold:

                    graph.add_edge(person_1, person_2,similarity=similarity)

        edge_count = graph.number_of_edges()

        print(threshold, edge_count)

        if 100 <= edge_count <= 1500:

            print("Selected similarity threshold:", threshold)

            return threshold

    print("Selected similarity threshold:", 0.3)

    return 0.3


MAX_IMAGE = 10
MAX_PEOPLE = 15

def draw_sample_person_graphs(graph, output_dir):

    person_list = list(graph.nodes())

    for image_index in range(MAX_IMAGE):

        center_person = person_list[image_index]

        sample_nodes = [center_person]

        for neighbor in graph.neighbors(center_person):

            sample_nodes.append(neighbor)

            if len(sample_nodes) == MAX_PEOPLE:
                break

        subgraph = graph.subgraph(sample_nodes)

        plt.figure(figsize=(12, 10))

        pos = nx.spring_layout(subgraph, seed=42)

        edge_widths = []

        for _, _, edge_data in subgraph.edges(data=True):

            similarity = edge_data["similarity"]

            edge_widths.append(similarity * 2)

        nx.draw_networkx_nodes(subgraph, pos, node_size=300)

        nx.draw_networkx_edges(subgraph, pos, width=edge_widths, alpha=0.7)

        labels = {}

        counter = 1

        for node in subgraph.nodes():

            labels[node] = f"Person{counter}"

            counter += 1

        nx.draw_networkx_labels(
            subgraph,
            pos,
            labels=labels,
            font_size=8
        )

        plt.title(f"Person Graph - Person1")

        plt.axis("off")

        plt.tight_layout()

        plt.savefig(output_dir /f"person_graph_{image_index + 1}.png", dpi=300)

        plt.close()

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    roi_files = sorted(ROI_DIR.glob("*.1D"))

    person_features = build_person_features(roi_files)

    similarity_threshold = (choose_similarity_threshold(person_features))

    person_graph = build_person_graph(person_features, similarity_threshold)

    save_person_graph_edges(person_graph, OUTPUT_DIR / "person_graph_edges.csv")

    save_graph_report(person_graph, similarity_threshold, OUTPUT_DIR / "graph_report.csv")

    draw_sample_person_graphs(person_graph, OUTPUT_DIR)

    print("Stage 2 completed.")

    print("Outputs saved in:", OUTPUT_DIR)

if __name__ == "__main__":

    main()

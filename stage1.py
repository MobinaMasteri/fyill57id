import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

DESKTOP = Path.home() / "Desktop"

ROI_DIR = DESKTOP / "rois_aal"
OUTPUT_DIR = DESKTOP / "stage1_outputs"

def read_fmri_file(file_path):
matrix = np.loadtxt(file_path)
return matrix

def build_correlation_matrix(matrix):
corr_matrix = np.corrcoef(matrix.T)
corr_matrix = np.nan_to_num(corr_matrix)
return corr_matrix

THRESHOLD_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7]
RANDOM_DATA = 100
def choose_threshold(roi_files):

random.seed(42)  

random_data_files = random.sample(roi_files, RANDOM_DATA)  

for threshold in THRESHOLD_VALUES:  

    edge_counts = []  

    for person_file in random_data_files:  

        matrix = read_fmri_file(person_file)  

        corr_matrix = build_correlation_matrix(matrix)  

        threshold_graph = build_threshold_graph(corr_matrix, threshold)  

        edge_counts.append(threshold_graph.number_of_edges())  

    average_edges = sum(edge_counts) / len(edge_counts)  

    print(threshold, average_edges)  

    if 100 <= average_edges <= 200:  

        print("Selected threshold:", threshold)  

        return threshold  

print("Selected threshold:", 0.5)  

return 0.5

def build_threshold_graph(corr_matrix, threshold):

n_rois = corr_matrix.shape[0]  

threshold_graph = nx.Graph()  

for roi in range(n_rois):  
    threshold_graph.add_node(roi)  

for i in range(n_rois):  
    for j in range(i + 1, n_rois):  

        corr = corr_matrix[i, j]  

        if abs(corr) >= threshold:  

            threshold_graph.add_edge(i, j, correlation=corr, distance=1 - abs(corr))  

return threshold_graph

def find_strongest_and_weakest(corr_matrix):

n_rois = corr_matrix.shape[0]  

strongest_pair = None  
weakest_pair = None  
strongest_value = -1  
weakest_value = 2  

for i in range(n_rois):  

    for j in range(i + 1, n_rois):  

        corr = corr_matrix[i, j]  

        abs_corr = abs(corr)  

        if abs_corr > strongest_value:  

            strongest_value = abs_corr  
            strongest_pair = (i, j, corr)  

        if (abs_corr < weakest_value) :  
            weakest_value = abs_corr  
            weakest_pair = (i, j, corr)  

return (strongest_pair, weakest_pair)

def save_edge_list(graph, output_path):

with open(output_path, "w", encoding="utf-8") as f:  

    f.write("first,second,correlation\n")  

    for first, second, cor in graph.edges(data=True):  

        f.write(f"{first},{second},"  
            f"{cor['correlation']:.3f}\n")

def save_person_report(person_dir, threshold_graph, selected_threshold, strongest_pair, weakest_pair):

report_path = (person_dir / "person_report.csv")  

with open( report_path, "w", encoding="utf-8") as f:  

    f.write("item,value\n")  

    f.write(f"selected_threshold,"  
        f"{selected_threshold}\n")  

    f.write(f"threshold_graph_edges,"  
        f"{threshold_graph.number_of_edges()}\n")  

    f.write(f"strongest_roi_1,"  
        f"{strongest_pair[0]}\n")  

    f.write(f"strongest_roi_2,"  
        f"{strongest_pair[1]}\n")  

    f.write(f"strongest_correlation,"  
        f"{strongest_pair[2]:.6f}\n")  

    f.write(f"weakest_roi_1,"  
        f"{weakest_pair[0]}\n")  

    f.write(f"weakest_roi_2,"  
        f"{weakest_pair[1]}\n")  

    f.write(f"weakest_correlation,"  
        f"{weakest_pair[2]:.6f}\n")  
      
    f.write(f"threshold_graph_density,"  
        f"{nx.density(threshold_graph):.6f}\n")

MAX_IMAGE = 10

def draw_threshold_graph(graph, output_path, title):
plt.figure(figsize=(16, 14))

pos = nx.spring_layout(graph, seed=42)  

nx.draw_networkx_nodes(graph, pos, node_size=30)  

edge_widths = []  

for first, second, edge_data in graph.edges(data=True):  
    corr_strength = abs(edge_data["correlation"])  

    edge_widths.append(corr_strength * 2)  

nx.draw_networkx_edges(graph, pos, width=edge_widths,alpha=0.6)  

plt.title(title)  

plt.axis("off")  

plt.tight_layout()  

plt.savefig(output_path, dpi=300)  

plt.close()

def process_person(file_path, image_counter, selected_threshold):

person_name = file_path.stem  

matrix = read_fmri_file(file_path)  

corr_matrix = build_correlation_matrix(matrix)  

threshold_graph = build_threshold_graph(corr_matrix, selected_threshold)  

strongest_pair, weakest_pair = (find_strongest_and_weakest(corr_matrix))  

person_dir = (OUTPUT_DIR / person_name)  

person_dir.mkdir(parents=True, exist_ok=True)  

save_edge_list(threshold_graph, person_dir / "threshold_graph_edges.csv")  

save_person_report(person_dir, threshold_graph, selected_threshold, strongest_pair, weakest_pair)  

if image_counter < MAX_IMAGE:  
    draw_threshold_graph(threshold_graph, person_dir / "threshold_graph.png", f"Threshold Graph - {person_name}")  
    image_counter += 1  

return image_counter

def main():

OUTPUT_DIR.mkdir(exist_ok=True)  

roi_files = sorted(ROI_DIR.glob("*.1D"))  

selected_threshold = (choose_threshold(roi_files))  

image_counter = 0  

for index, file_path in enumerate(roi_files,start=1):  

    print(f"[{index}/{len(roi_files)}] "  
          f"Processing {file_path.name}")  

    image_counter = (process_person(file_path, image_counter, selected_threshold))  

print("Stage 1 completed.")  

print("Selected threshold:", selected_threshold)  

print("Outputs saved in:", OUTPUT_DIR)

if name == "main":
main()

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import networkx as nx
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv
import os
import matplotlib.pyplot as plt

# Knee X-ray organ-to-abnormality mapping
knee_organ_to_abnormalities = {
    "Medial Condyle": ["Joint Space Narrowing", "Osteophytes", "Tibial Spikes"],
    "Lateral Condyle": ["Joint Space Narrowing", "Osteophytes", "Tibial Spikes"],
    "Tibiofemoral Joint": ["Joint Space Narrowing", "Osteophytes", "Effusion", "Tibial Spikes"],
    "Patellofemoral Joint": ["Joint Space Narrowing", "Osteophytes"],
    "Tibia": ["Fracture", "Spikes"],
    "Femur": ["Fracture", "Osteophytes"],
    "Patella": ["Fracture", "Dislocation"]
}

# Reverse Mapping: Abnormalities to Primary Organ
knee_abnormality_to_organ = {abnormality.lower(): organ
                             for organ, abnormalities in knee_organ_to_abnormalities.items()
                             for abnormality in abnormalities}

presence_encoding = {"Present": 1.0, "Absent": 0.0}

def extract_findings(report):
    findings = []
    sentences = report.split(",")

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        clean_sentence = re.sub(r"\s+", " ", sentence.lower())
        presence = "Absent" if any(neg in clean_sentence for neg in ["no ","not seen","absent","normal"]) else "Present"

        found_abnormalities = set()
        found_organ = None

        for organ in knee_organ_to_abnormalities.keys():
            if re.search(rf"\b{re.escape(organ.lower())}\b", clean_sentence):
                found_organ = organ
                break

        for abnormality in knee_abnormality_to_organ.keys():
            pattern = rf"\b{re.escape(abnormality)}\b"
            if re.search(pattern, clean_sentence):
                found_abnormalities.add(abnormality)

        for abnormality in found_abnormalities:
            organ = found_organ if found_organ else knee_abnormality_to_organ[abnormality]
            findings.append({"Organ": organ, "Abnormality": abnormality, "Presence": presence})

    return findings

def create_scene_graph(findings):
    G = nx.DiGraph()

    for finding in findings:
        organ = finding["Organ"]
        abnormality = finding["Abnormality"]
        presence = finding["Presence"]

        G.add_node(organ, type="Organ")
        G.add_node(abnormality, type="Abnormality")
        G.add_edge(organ, abnormality, presence=presence)

    return G

def save_scene_graph(G, save_path):
    """Save the scene graph as a PNG image."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    labels = {node: node for node in G.nodes()}
    nx.draw(G, pos, with_labels=True, node_color="skyblue", edge_color="gray", node_size=2000, font_size=10)
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    plt.savefig(save_path)
    plt.close()

def convert_to_pyg(G):
    node_to_idx = {node: i for i, node in enumerate(G.nodes())}
    edge_index = []
    node_features = []

    for node in G.nodes():
        node_type = G.nodes[node]["type"]
        type_vector = [1, 0] if node_type == "Organ" else [0, 1]

        presence_value = 0.0
        for neighbor in G[node]:
            presence_value = presence_encoding[G[node][neighbor]["presence"]]

        node_features.append(type_vector + [presence_value])

    for u, v in G.edges():
        edge_index.append([node_to_idx[u], node_to_idx[v]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
    node_features = torch.tensor(node_features, dtype=torch.float) if node_features else torch.empty((0, 3), dtype=torch.float)

    return Data(x=node_features, edge_index=edge_index), node_to_idx

class SceneGraphGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SceneGraphGAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=2)
        self.conv2 = GATConv(hidden_channels * 2, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

def pad_graph_embeddings(embeddings, max_nodes):
    """ Pads node embeddings to ensure uniform shape across batch """
    pad_size = max_nodes - embeddings.shape[0]
    if pad_size > 0:
        padding = torch.zeros((pad_size, embeddings.shape[1]), dtype=embeddings.dtype)
        embeddings = torch.cat([embeddings, padding], dim=0)
    return embeddings

def process_batch(ip, reports, max_nodes=None):
    model = SceneGraphGAT(in_channels=3, hidden_channels=8, out_channels=50)
    all_graphs = []
    num_nodes_per_graph = []
    path = "./scenegraphs"
    for report in reports:
        findings = extract_findings(report)
        #print(findings)
        G = create_scene_graph(findings)
        save_scene_graph(G,path)
        graph_data, _ = convert_to_pyg(G)
        all_graphs.append(graph_data)
        num_nodes_per_graph.append(graph_data.x.shape[0])

    max_nodes = max_nodes or max(num_nodes_per_graph)

    batch_graphs = Batch.from_data_list(all_graphs)

    with torch.no_grad():
        batch_embeddings = model(batch_graphs.x, batch_graphs.edge_index)

    split_embeddings = batch_embeddings.split(num_nodes_per_graph, dim=0)
    padded_embeddings = torch.stack([pad_graph_embeddings(emb, max_nodes) for emb in split_embeddings])

    return padded_embeddings

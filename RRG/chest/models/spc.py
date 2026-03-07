import networkx as nx
import spacy
import numpy as np
import gensim.downloader as api
import torch
import re
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
import matplotlib.patches as mpatches



# Organ-Abnormality Mapping
organ_to_abnormalities = {
    "Right Lung": ["Atelectasis", "Pneumothorax", "Pleural Effusion", "Consolidation", "Fibrosis",
                   "Pneumonia", "Interstitial Lung Disease",
                   "Hyperinflation", "Nodule", "Tumor", "Granuloma",
                   "Edema", "Calcifications", "Nodular Opacity","opacities","pulmonary vascularity", "opacification"],
    "Left Lung": ["Atelectasis", "Pneumothorax", "Pleural Effusion", "Consolidation", "Fibrosis",
                  "Pneumonia", "Interstitial Lung Disease",
                  "Hyperinflation", "Nodule", "Tumor", "Granuloma",
                  "Edema", "Calcifications", "Nodular Opacity"],
    "Lungs": ["clear","Diffuse interstitial","focal airspace", "hyperexpanded","focal infiltrates", "infiltrates", "pulmonary nodules", "pulmonary"],
    "Trachea": ["Stenosis", "Tracheomalacia", "Airway Obstruction"],
    "Bronchus": ["Bronchiectasis"],
    "Alveoli": ["Pulmonary Edema", "Emphysema"],
    "Heart": ["enlarged","vascular congestion", "Cardiomegaly", "Hypertension", "Cardiopulmonary Congestion", "Edema","cardiomediastinal silhouette","Cardiac and mediastinal contours", "Cardiac silhouette"],
    "Right Main Bronchus": ["Bronchiectasis", "Airway Obstruction"],
    "Left Main Bronchus": ["Bronchiectasis", "Airway Obstruction"],
    "Aorta": ["Atherosclerosis", "Aneurysm", "Vascular Calcifications", "Hypertension", "atherosclerotic","tortuous"],
    "Pulmonary Artery": ["Pulmonary Embolism", "Pulmonary Hypertension", "engorged"],
    "Subclavian Artery": ["Thrombosis"],
    "Esophagus": ["Stricture", "Dilation"],
    "Stomach": ["Hernia"],
    "Liver": ["Hepatomegaly", "Granuloma", "Scarring"],
    "Gallbladder": ["Gallstones"],
    "Pancreas": ["Pancreatitis", "Mass", "Cyst"],
    "Large Intestine": ["Diverticulosis", "Bowel Obstruction"],
    "Rib": ["Fractures", "Osteoporosis","Fracture"],
    "Sternum": ["Fracture","Fractures"],
    "Clavicle": ["Fracture", "Dislocation","Fractures"],
    "Scapula": ["Fracture", "Dislocation","Fractures"],
    "Cervical Vertebrae": ["Fracture", "Spondylosis", "Degenerative Changes","Fractures"],
    "Thoracic Vertebrae": ["Fracture", "Scoliosis", "Kyphosis", "Spondylosis", "Degenerative Changes","Fractures"],
    "Spine": ["Fracture", "Spondylosis", "Degenerative Changes","Fractures"],
    "Pelvis": ["Fracture", "Osteopenia","Fractures"],
    "Femur": ["Fracture", "Dislocation","Fractures"],
    "Humerus": ["Fracture", "Dislocation","Fractures"],
    "Sacrum": ["Fracture","Fractures"],
    "Patella": ["Fracture", "Dislocation"],
    "Spinal Cord": ["Compression", "Injury"],
    "Nerve Roots": ["Compression", "Radiculopathy"],
    "Lymph Nodes": ["Lymphadenopathy"],
    "Thymus": ["Tumor"],
    "Diaphragm": ["Hernia", "Eventration", "Paralysis"],
    "Mediastinum": ["Pneumomediastinum"],
    "Pericardium": ["Constrictive Pericarditis"],
    "Hilum": ["Hilar Lymphadenopathy"],
    "Hemidiaphragm": ["Elevation", "Paralysis","obscuration"],
    "Apices": ["Pancoast Tumor"],
    "pulmonary venous":["engorgement"]
}

# Reverse Mapping: Abnormalities to Primary Organ
# Reverse Mapping: Abnormalities to Primary Organ
abnormality_to_organ = {re.sub(r"\s+", " ", abnormality.lower()): organ 
                        for organ, abnormalities in organ_to_abnormalities.items() 
                        for abnormality in abnormalities}

presence_encoding = {"Present": 1.0, "Absent": 0.0}

def extract_findings(report):
    findings = []
    sentences = report.split(".")

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        clean_sentence = re.sub(r"\s+", " ", sentence.lower())

        presence = "Absent" if any(neg in clean_sentence for neg in ["no", "not seen", "absent", "No evidence","free"]) else "Present"

        found_abnormalities = set()
        found_organ = None

        # Detect explicit organs
        for organ in organ_to_abnormalities.keys():
            if re.search(rf"\b{re.escape(organ.lower())}\b", clean_sentence):
                found_organ = organ
                break

        if re.search(r"\b(no|clear|normal|unremarkable|large|developed|low)\b", clean_sentence):
            if found_organ:
                findings.append({"Organ": found_organ, "Abnormality": "Clear/Normal", "Presence": "Present"})
                continue
        # Detect abnormalities
        for abnormality in abnormality_to_organ.keys():
            pattern = r"\b" + re.escape(abnormality) + r"\b"
            if re.search(pattern, clean_sentence):
                found_abnormalities.add(abnormality)

        # Assign findings
        for abnormality in found_abnormalities:
            organ = found_organ if found_organ else abnormality_to_organ[abnormality]
            findings.append({"Organ": organ, "Abnormality": abnormality, "Presence": presence})

    return findings

def create_scene_graph(findings):
    G = nx.DiGraph()  # Directed Graph

    for finding in findings:
        organ = finding["Organ"]
        abnormality = finding["Abnormality"]
        presence = finding["Presence"]

        G.add_node(organ, type="Organ")
        G.add_node(abnormality, type="Abnormality")

        G.add_edge(organ, abnormality, presence=presence)
    
    if G.number_of_nodes() == 0:
        G.add_node("Lungs", type="Organ")
        G.add_node("Clear/Normal", type="Abnormality")
        G.add_edge("Lungs", "Clear/Normal", presence="Present")

    return G

def convert_to_pyg(G):
    node_to_idx = {node: i for i, node in enumerate(G.nodes())}
    edge_index = []
    node_features = []

    for node in G.nodes():
        node_type = G.nodes[node]["type"]  
        type_vector = [1, 0] if node_type == "Organ" else [0, 1]  # One-hot encoding

        presence_value = 0.0  # Default Absent
        for neighbor in G[node]:  
            presence_value = presence_encoding[G[node][neighbor]["presence"]]

        node_features.append(type_vector + [presence_value])

    for u, v in G.edges():
        edge_index.append([node_to_idx[u], node_to_idx[v]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    node_features = torch.tensor(node_features, dtype=torch.float)

    return Data(x=node_features, edge_index=edge_index), node_to_idx

class QKVGraphAttentionLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(QKVGraphAttentionLayer, self).__init__(aggr='add')
        self.q_proj = nn.Linear(in_channels, out_channels)
        self.k_proj = nn.Linear(in_channels, out_channels)
        self.v_proj = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        Q = self.q_proj(x)  # [N, out_channels]
        K = self.k_proj(x)
        V = self.v_proj(x)

        return self.propagate(edge_index, Q=Q, K=K, V=V)

    def message(self, Q_i, K_j, V_j, index, ptr, size_i):
        scale = Q_i.size(-1) ** 0.5
        attention_scores = (Q_i * K_j).sum(dim=-1) / scale  # [E]
        attention_weights = softmax(attention_scores, index)  # Normalize over neighbors
        return attention_weights.unsqueeze(-1) * V_j

    def update(self, aggr_out):
        return aggr_out

class SceneGraphGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SceneGraphGAT, self).__init__()
        self.attn1 = QKVGraphAttentionLayer(in_channels, hidden_channels)
        self.attn2 = QKVGraphAttentionLayer(hidden_channels, out_channels)

    def forward(self, node_d, edge_d):
        x = F.elu(self.attn1(node_d, edge_d))  # First QKV attention
        x = self.attn2(x, edge_d)              # Second QKV attention
        return x  

def process_report(report):
    findings = extract_findings(report)
    G = create_scene_graph(findings)
    graph_data, node_to_idx = convert_to_pyg(G)

    model = SceneGraphGAT(in_channels=3, hidden_channels=8, out_channels=50)

    with torch.no_grad():
        node_embeddings = model(graph_data.x, graph_data.edge_index)

    return node_embeddings,findings,G

def process_batch(image_paths,reports):
    all_embeddings = []
    for image_path, report in zip(image_paths, reports):
        # Generate and save scene graph
        embeddings,findings,G = process_report(report)
        all_embeddings.append(embeddings)

    max_nodes = max(emb.shape[0] for emb in all_embeddings)  # Find max number of nodes in any graph
    embedding_dim = all_embeddings[0].shape[1]  # Get embedding dimension

    padded_embeddings = []
    for emb in all_embeddings:
        num_nodes = emb.shape[0]
        if num_nodes < max_nodes:
            pad = torch.zeros((max_nodes - num_nodes, embedding_dim))  # Padding with zeros
            emb = torch.cat([emb, pad], dim=0)
        padded_embeddings.append(emb)

    # Stack into a single tensor
    batch_embeddings = torch.stack(padded_embeddings)  # Shape: (batch_size, max_nodes, embedding_dim)
    return batch_embeddings




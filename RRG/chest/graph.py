import networkx as nx
import matplotlib.pyplot as plt
import spacy

# Load NLP model
nlp = spacy.load("en_core_web_sm")

def xray_report_to_scenegraph(report):
    """Convert a chest X-ray report into a scene graph (Organ-Abnormality & Organ-Organ Relations)."""
    doc = nlp(report)
    G = nx.DiGraph()

    organ_nodes = set()
    abnormality_nodes = set()
    edges = []

    for token in doc:
        # Identify organs (anatomical structures)
        if token.pos_ == "NOUN":
            G.add_node(token.text, type="organ")
            organ_nodes.add(token.text)

        # Identify abnormalities (findings)
        elif token.pos_ == "ADJ" and token.head.pos_ == "NOUN":
            G.add_node(token.text, type="abnormality")
            G.add_edge(token.head.text, token.text, relation="has_finding")
            abnormality_nodes.add(token.text)
            edges.append((token.head.text, token.text, "has_finding"))


    return G, organ_nodes, abnormality_nodes, edges

def plot_scene_graph(graph, organ_nodes, abnormality_nodes, edges, title="Scene Graph for X-ray Report", save_path="scene_graph.png"):
    """Plot the generated scene graph with clearly visible relations and save it to a file."""
    plt.figure(figsize=(10, 7))

    # Improve layout spacing
    pos = nx.spring_layout(graph, k=1.0, seed=42)  # Increase spacing between nodes

    # Extract labels
    labels = {node: node for node in graph.nodes()}
    edge_labels = { (u, v): rel for u, v, rel in edges }

    # Color nodes: Organs (blue), Abnormalities (red), Negations (gray)
    node_colors = []
    for node in graph.nodes():
        if node in organ_nodes:
            node_colors.append("skyblue")  # Organ
        elif node in abnormality_nodes:
            node_colors.append("lightcoral")  # Abnormality

    # Draw nodes with colors
    nx.draw(graph, pos, with_labels=True, node_size=3000, node_color=node_colors, edge_color="black", linewidths=1.5, font_size=10, font_weight="bold")

    # Draw edges with thickness
    nx.draw_networkx_edges(graph, pos, edge_color="black", width=2.5, alpha=0.7)

    # Draw edge labels
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=9, font_color="darkred")

    plt.title(title, fontsize=14, fontweight="bold")

    # Save the figure instead of showing it
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Graph saved to {save_path}")
    plt.close()

# Example usage
if __name__ == "__main__":
    example_report = "The cardiomediastinal silhouette is normal in size and contour. The heart and lungs appear unremarkable. There are a few opacities in the lung bases bilaterally.definitive No pneumothorax or pleural effusion. Displaced fracture of the mid one-third of the right clavicle."
    
    scene_graph, organ_nodes, abnormality_nodes, edges = xray_report_to_scenegraph(example_report)
    plot_scene_graph(scene_graph, organ_nodes, abnormality_nodes, edges)


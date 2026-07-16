import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

def generate_pd_network():
    # Load interactions
    df_edges = pd.read_csv('interactions.csv')
    
    # Load gene roles (nodes)
    df_nodes = pd.read_csv('pd_genes.csv')
    
    # Create graph
    G = nx.from_pandas_edgelist(df_edges, source='source', target='target')
    
    # Draw graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color='black')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')
    
    plt.title("Parkinson's Disease Protein Interaction Network")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_pd_network()
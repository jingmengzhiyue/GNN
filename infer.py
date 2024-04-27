from model import EnhancedGAT
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
import torch_geometric.utils as pyg_utils
from torch_geometric.utils import to_networkx
import networkx as nx
import pandas as pd
import torch 

data_pth = "dataset.xlsx"
sheet_name='Token'
df = pd.read_excel(data_pth, sheet_name=sheet_name)

expanded_rows = []  # List to hold expanded rows
for index, row in df.iterrows():
    tokens = row['翻译Token'].split()
    for token in tokens:
        expanded_rows.append({'原文Token': row['原文Token'], '翻译Token': token})

# Create a new DataFrame from the expanded rows
expanded_df = pd.DataFrame(expanded_rows)

origin = expanded_df['原文Token'].tolist()
trans = expanded_df['翻译Token'].tolist()
edges = list(zip(origin, trans))

nodes = list(set(origin + trans))
node_index = {node: idx for idx, node in enumerate(nodes)}
# edge index
edge_index = torch.tensor([[node_index[src], node_index[dst]] for src, dst in edges if src in node_index and dst in node_index], dtype=torch.long).t().contiguous()


# Node feature, using simple one-hot coding here
features = torch.eye(len(node_index))

# Create a graph data object
data = Data(x=features, edge_index=edge_index)

# Convert to NetworkX diagram
G = to_networkx(data, to_undirected=True)

# visual settings
plt.figure(figsize=(16, 8))  
pos = nx.spring_layout(G)  

# 绘制图
nx.draw(G, pos,
        with_labels=True, 
        labels={idx: node for node, idx in node_index.items()},  
        node_color='skyblue',  
        node_size=50,  
        edge_color='black',  
        linewidths=2,  
        font_size=10,  
        font_color='darkred')  
# plt.savefig("origin.png")

device = "cuda" if torch.cuda.is_available() else "cpu"



model = EnhancedGAT(num_features=len(node_index), hidden_channels=8, num_classes=16)
model.load_state_dict(torch.load('EnhancedGAT.pt'))
_, node_embeddings = model(data.x, data.edge_index)
node_embeddings = node_embeddings.detach()
# for i, emb in enumerate(node_embeddings):
#     print(f"Node {nodes[i]} Embedding: {emb.numpy()}")


tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(node_embeddings.numpy())  # Convert to NumPy array for Scikit-Learn

plt.figure(figsize=(10, 8))
for i, label in enumerate(nodes):
    x, y = embeddings_2d[i]
    plt.scatter(x, y)
    plt.annotate(label, (x, y), textcoords="offset points", xytext=(0,10), ha='center')
plt.title('2D Visualization of Node Embeddings')
# plt.show()
# plt.savefig("embedding.png")

def find_nodes_related_to_group(input_nodes, node_index, node_embeddings, nodes, n=1):
    input_indices = []
    """
    Identifies nodes related to a given set of input nodes within a graph.
    
    Parameters:
    - input_nodes: A list of input nodes.
    - node_index: A dictionary mapping nodes to their respective indices.
    - node_embeddings: Embedding vectors for each node in the graph.
    - nodes: A list of node names corresponding to node_embeddings.
    - n: The number of top related nodes to return, defaults to 1.
    
    Returns:
    - A list of the top n nodes most related to the input nodes.
    """
    for node in input_nodes:
        if node in node_index:
            input_indices.append(node_index[node])
        else:
            print(f"Node {node} is not found in the diagram.")
            return []

    if not input_indices:
        print("No valid input node...")
        return []

    # Get the embeddings vector of all input nodes and calculate the average vector
    input_embeddings = node_embeddings[input_indices]
    average_embedding = input_embeddings.mean(dim=0)

    # Calculate the cosine similarity of all nodes to the average embeddings vector
    cos_sim = F.cosine_similarity(node_embeddings, average_embedding.unsqueeze(0), dim=1)

    # Get the indexes of the n nodes with the highest similarity
    _, top_n_indices = torch.topk(cos_sim, n + len(input_indices))  # Add input nodes to prevent yourself from being selected

    # Find the corresponding node name through the index and filter out the input node
    related_nodes = [nodes[idx] for idx in top_n_indices if idx not in input_indices][:n]

    return related_nodes


input_nodes1 = ["Dgjcxkqt", "Dgjcxkqtcn", "Dgjcxkqwt"]  
n = 2  
related_nodes = find_nodes_related_to_group(input_nodes1, node_index, node_embeddings, nodes, n)
print(f"与{input_nodes1}最相关的{n}个节点是: {related_nodes}")


input_nodes2 = ["Fgwvuejg", "Fgwvuejgu", "Fgwvuejgp", "Fgwvuejgt"]  
n = 2  
related_nodes = find_nodes_related_to_group(input_nodes2, node_index, node_embeddings, nodes, n)
print(f"与{input_nodes2}最相关的{n}个节点是: {related_nodes}")


input_nodes3 = ["Ykuugpuejchv", "Ykuugpuejchvgu", "Ykuugpuejchvgp"]  
n = 2  
related_nodes = find_nodes_related_to_group(input_nodes3, node_index, node_embeddings, nodes, n)
print(f"与{input_nodes3}最相关的{n}个节点是: {related_nodes}")
import pandas as pd
import torch 
from torch_geometric.data import Data
import pickle

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


# Node characteristics, using simple one-hot encoding
features = torch.eye(len(node_index))

# Create a graph data object
data = Data(x=features, edge_index=edge_index)

print(len(node_index))

with open('data.pkl', 'wb') as f:
    pickle.dump(data, f)
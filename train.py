from model import EnhancedGAT
import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from loguru import logger
import torch.nn.functional as F

logger.add("training_logs.log", format="{time} {level} {message}", level="INFO")

device = "cuda" if torch.cuda.is_available() else "cpu"

data_pth = "dataset.xlsx"
sheet_name = "Token"
df = pd.read_excel(data_pth, sheet_name=sheet_name)

expanded_rows = []  # List to hold expanded rows
for index, row in df.iterrows():
    tokens = row["翻译Token"].split()
    for token in tokens:
        expanded_rows.append({"原文Token": row["原文Token"], "翻译Token": token})

# Create a new DataFrame from the expanded rows
expanded_df = pd.DataFrame(expanded_rows)

origin = expanded_df["原文Token"].tolist()
trans = expanded_df["翻译Token"].tolist()
edges = list(zip(origin, trans))

nodes = list(set(origin + trans))
node_index = {node: idx for idx, node in enumerate(nodes)}
# 边索引
edge_index = (
    torch.tensor(
        [
            [node_index[src], node_index[dst]]
            for src, dst in edges
            if src in node_index and dst in node_index
        ],
        dtype=torch.long,
    )
    .t()
    .contiguous()
)


features = torch.eye(len(node_index))


data = Data(x=features, edge_index=edge_index)


model = EnhancedGAT(num_features=537, hidden_channels=8, num_classes=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
model.to(device)
data.to(device)


def compute_loss(pos_output, neg_output, margin=1.0):
    pos_dist = torch.norm(pos_output[0] - pos_output[1], p=2)
    neg_dist = torch.norm(pos_output[0] - neg_output, p=2)
    return F.relu(pos_dist - neg_dist + margin)


def train():

    model.train()
    optimizer.zero_grad()
    _, out = model(data.x, data.edge_index)

    loss = 0
    for edge in edges:
        src, dst = node_index[edge[0]], node_index[edge[1]]
        neg_dst = np.random.choice(
            [i for i in range(len(node_index)) if i not in [src, dst]]
        )

        loss += compute_loss((out[src], out[dst]), out[neg_dst])
    loss /= len(edges)  #
    loss.backward()
    optimizer.step()
    return loss.item()


# 进行训练
for epoch in range(2000):
    loss = train()
    # print(f'Epoch {epoch+1}, Loss: {loss}')
    logger.info(f"Epoch {epoch+1}, Loss: {loss}")
    if loss < 1e-5:
        break

# save Model
save_path = "EnhancedGAT.pt"
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")

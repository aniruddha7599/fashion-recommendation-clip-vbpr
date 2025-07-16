import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
df_path = 'C:/Users/Aniruddha shinde/DL Project/Local/preprocessed_fashion_data.pkl'
user_item_path = 'C:/Users/Aniruddha shinde/DL Project/Local/user_item_interactions.pkl'
save_dir = 'C:/Users/Aniruddha shinde/DL Project/Local/clip_fashion_model'

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df = pd.read_pickle(df_path)
with open(user_item_path, 'rb') as f:
    user_item_data = pickle.load(f)

# Convert user_item_data to DataFrame and aggregate duplicates
rows = []
for user_id, interactions in user_item_data.items():
    for item_id, description, image_paths in interactions:
        rows.append({'user_id': user_id, 'item_id': item_id, 'description': description, 'image_paths': image_paths})
user_item_df = pd.DataFrame(rows)
user_item_df = user_item_df.groupby(['user_id', 'item_id']).agg({
    'description': 'first',
    'image_paths': lambda x: list(set(sum(x, [])))
}).reset_index()

# Map user IDs to indices
user_mapping = {user_id: idx for idx, user_id in enumerate(user_item_df['user_id'].unique())}
user_item_df['user_idx'] = user_item_df['user_id'].map(user_mapping)

# Map item IDs to df indices using 'asin'
item_mapping = dict(zip(df['asin'], df.index))
user_item_df['item_idx'] = user_item_df['item_id'].map(item_mapping)

# VBPR Model
class VBPR(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=100, visual_dim=512):
        super(VBPR, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.visual_embeddings = nn.Linear(visual_dim, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(num_items))
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)
        nn.init.xavier_uniform_(self.visual_embeddings.weight)
        nn.init.zeros_(self.visual_embeddings.bias)

    def forward(self, user_idx, item_idx, image_features):
        user_emb = self.user_embeddings(user_idx)
        item_emb = self.item_embeddings(item_idx)
        batch_size = user_idx.size(0)
        image_features = image_features.unsqueeze(0).expand(batch_size, -1, -1)
        visual_emb = self.visual_embeddings(image_features)
        scores = (user_emb.unsqueeze(1) * (item_emb + visual_emb)).sum(-1) + self.bias[item_idx]
        return scores

# Load saved models and embeddings
processor = CLIPProcessor.from_pretrained(save_dir, use_fast=False)
model = CLIPModel.from_pretrained(save_dir).to(device)
model.eval()

num_users = len(user_mapping)
num_items = len(df)
vbpr_model = VBPR(num_users, num_items).to(device)
vbpr_model.load_state_dict(torch.load(f"{save_dir}/vbpr_model.pt", weights_only=True))
vbpr_model.eval()

all_image_features = torch.load(f"{save_dir}/all_image_features.pt", weights_only=True)

# Evaluation
users_to_evaluate = np.random.choice(list(user_mapping.values()), size=100, replace=False)
categories = ["shirt", "pants", "dress", "shoes", "jacket", "necklace", "hat", "bag", "belt"]
item_indices = list(range(len(df)))
item_tensor = torch.tensor(item_indices, dtype=torch.long).to(device)

# Precompute text features for categories
category_features = {}
for category in categories:
    inputs = processor(text=[category], return_tensors="pt", padding="max_length", max_length=77, truncation=True)
    with torch.no_grad():
        features = model.get_text_features(**{k: v.to(device) for k, v in inputs.items()})
    category_features[category] = features / features.norm(dim=-1, keepdim=True)

# Normalize all_image_features
all_image_features = all_image_features / all_image_features.norm(dim=-1, keepdim=True)

# Move static tensors to GPU once
all_image_features = all_image_features.to(device)
item_tensor = item_tensor.to(device)

# Batch users for evaluation
batch_size = 10
user_batches = [users_to_evaluate[i:i + batch_size] for i in range(0, len(users_to_evaluate), batch_size)]

category_metrics = {category: {'recall': [], 'ndcg': []} for category in categories}

for batch_idx, user_batch in enumerate(tqdm(user_batches, desc="Evaluating user batches")):
    user_tensor = torch.tensor(user_batch, dtype=torch.long).to(device)
    batch_size = len(user_batch)
    
    # Get relevant items for all users in the batch
    labels = torch.zeros(batch_size, len(item_indices), dtype=torch.float32).to(device)
    for idx, user_idx in enumerate(user_batch):
        user_items = user_item_df[user_item_df['user_idx'] == user_idx]['item_idx'].values
        user_items_tensor = torch.tensor(user_items, dtype=torch.long).to(device)
        labels[idx, user_items_tensor] = 1.0
    
    # Compute base VBPR scores for all items for each user in the batch
    with torch.no_grad():
        base_scores = vbpr_model(user_tensor, item_tensor, all_image_features)
    
    # Track recommended items to apply diversity penalty
    recommended_items = set()
    
    # Compute metrics for each category
    for category in categories:
        query_features = category_features[category]
        similarity_scores = torch.matmul(all_image_features, query_features.T).squeeze(-1)
        similarity_scores = similarity_scores.unsqueeze(0).expand(batch_size, -1)
        
        # Debug similarity scores
        print(f"Category: {category}, similarity_scores mean: {similarity_scores.mean().item():.4f}, min: {similarity_scores.min().item():.4f}, max: {similarity_scores.max().item():.4f}")
        
        # Combine VBPR scores with category similarity
        weight = 3.0  # Increased to 3.0 to boost category influence
        combined_scores = base_scores + weight * similarity_scores
        
        # Apply diversity penalty
        penalty = torch.zeros_like(combined_scores)
        for item_idx in recommended_items:
            penalty[:, item_idx] = -0.5  # Reduce score for already recommended items
        combined_scores += penalty
        
        top_k_indices = combined_scores.topk(5, dim=1).indices
        
        # Add recommended items to the set
        for idx in top_k_indices.flatten().tolist():
            recommended_items.add(idx)
        
        # Debug shapes and devices
        print(f"labels shape: {labels.shape}, device: {labels.device}")
        print(f"top_k_indices shape: {top_k_indices.shape}, device: {top_k_indices.device}")
        
        # Print top 5 indices for batches 0, 1, and 2
        if batch_idx in [0, 1, 2]:
            print(f"Batch {batch_idx}, Category: {category}, Top 5 indices: {top_k_indices[0].tolist()}")
        
        top_k_labels = torch.gather(labels, 1, top_k_indices)
        
        relevant_counts = labels.sum(dim=1)
        retrieved_relevants = top_k_labels.sum(dim=1)
        recalls = retrieved_relevants / relevant_counts
        recalls[relevant_counts == 0] = 0.0
        
        ideal_dcg = torch.tensor([sum(1 / np.log2(np.arange(2, 6))) for _ in range(batch_size)]).to(device)
        dcg = (top_k_labels / torch.log2(torch.arange(2, 7, device=device))).sum(dim=1)
        ndcgs = dcg / ideal_dcg
        ndcgs[relevant_counts == 0] = 0.0
        
        category_metrics[category]['recall'].extend(recalls.tolist())
        category_metrics[category]['ndcg'].extend(ndcgs.tolist())

# Print final metrics
for category in categories:
    recalls = category_metrics[category]['recall']
    ndcgs = category_metrics[category]['ndcg']
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
    print(f"Category '{category}': Average Recall@5={avg_recall:.4f}, Average NDCG@5={avg_ndcg:.4f}")

overall_recall = sum([r for cat in category_metrics.values() for r in cat['recall']]) / (len(categories) * len(users_to_evaluate))
overall_ndcg = sum([n for cat in category_metrics.values() for n in cat['ndcg']]) / (len(categories) * len(users_to_evaluate))
print(f"Overall Average Recall@5: {overall_recall:.4f}")
print(f"Overall Average NDCG@5: {overall_ndcg:.4f}")
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import CLIPProcessor, CLIPModel
import pickle
import os

# Debug: Print transformers version
st.write(f"Transformers version: {transformers.__version__}")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
df_path = 'C:/Users/Aniruddha shinde/DL Project/Local/preprocessed_fashion_data.pkl'
user_item_path = 'C:/Users/Aniruddha shinde/DL Project/Local/user_item_interactions.pkl'
save_dir = 'C:/Users/Aniruddha shinde/DL Project/Local/clip_fashion_model'

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
model = CLIPModel.from_pretrained(save_dir)

# Move CLIP model to device using to_empty if needed
try:
    model = model.to(device)
except NotImplementedError:
    model = model.to_empty(device=device)
model.eval()

num_users = len(user_mapping)
num_items = len(df)

# Initialize VBPR model and move to device using to_empty
vbpr_model = VBPR(num_users, num_items)
try:
    vbpr_model = vbpr_model.to(device)
except NotImplementedError:
    vbpr_model = vbpr_model.to_empty(device=device)

# Load state dict and set to eval mode
vbpr_model.load_state_dict(torch.load(f"{save_dir}/vbpr_model.pt", weights_only=True))
vbpr_model.eval()

all_image_features = torch.load(f"{save_dir}/all_image_features.pt", weights_only=True)

# Normalize all_image_features
all_image_features = all_image_features / all_image_features.norm(dim=-1, keepdim=True)

# Move to device
all_image_features = all_image_features.to(device)
item_indices = list(range(len(df)))
item_tensor = torch.tensor(item_indices, dtype=torch.long).to(device)

# Precompute text features for categories with more specific prompts
category_prompts = {
    "shirt": "men’s cotton shirt",
    "pants": "men’s denim pants",
    "dress": "women’s summer dress",
    "shoes": "men’s leather shoes",
    "jacket": "men’s leather jacket",
    "necklace": "women’s silver necklace",
    "hat": "men’s baseball hat",
    "bag": "women’s leather handbag",
    "belt": "men’s leather belt"
}
category_features = {}
for category, prompt in category_prompts.items():
    inputs = processor(text=[prompt], return_tensors="pt", padding="max_length", max_length=77, truncation=True)
    with torch.no_grad():
        features = model.get_text_features(**{k: v.to(device) for k, v in inputs.items()})
        norm = features.norm(dim=-1, keepdim=True)
        features = features / norm
    category_features[category] = features
    st.write(f"Debug: Category '{category}' feature norm after normalization: {category_features[category].norm(dim=-1).item():.4f}")

# Streamlit app
st.title("Fashion Recommendation System")

# User selection
user_ids = list(user_mapping.keys())
selected_user = st.selectbox("Select a user:", user_ids)
user_idx = user_mapping[selected_user]
user_tensor = torch.tensor([user_idx], dtype=torch.long).to(device)

# Category selection
selected_categories = st.multiselect("Select categories:", list(category_prompts.keys()), default=["shirt"])

# Recommend items
if st.button("Get Recommendations"):
    recommended_items = set()  # For diversity penalty
    for category in selected_categories:
        st.subheader(f"Recommendations for {category.capitalize()}:")
        
        # Compute VBPR scores
        with torch.no_grad():
            base_scores = vbpr_model(user_tensor, item_tensor, all_image_features)
        
        # Normalize VBPR scores to range [-1, 1]
        base_scores_min = base_scores.min()
        base_scores_max = base_scores.max()
        if base_scores_max > base_scores_min:
            base_scores = 2 * (base_scores - base_scores_min) / (base_scores_max - base_scores_min) - 1
        else:
            base_scores = torch.zeros_like(base_scores)
        
        # Debug VBPR scores
        st.write(f"Debug: VBPR scores after normalization - mean: {base_scores.mean().item():.4f}, min: {base_scores.min().item():.4f}, max: {base_scores.max().item():.4f}")
        
        # Compute category similarity scores
        query_features = category_features[category]
        similarity_scores = torch.matmul(all_image_features, query_features.T).squeeze(-1)
        similarity_scores = similarity_scores.unsqueeze(0)
        
        # Debug similarity scores
        st.write(f"Debug: Category '{category}' similarity scores - mean: {similarity_scores.mean().item():.4f}, min: {similarity_scores.min().item():.4f}, max: {similarity_scores.max().item():.4f}")
        
        # Combine scores with increased weight = 75.0
        weight = 75.0
        combined_scores = base_scores + weight * similarity_scores
        
        # Apply diversity penalty
        penalty = torch.zeros_like(combined_scores)
        for item_idx in recommended_items:
            penalty[:, item_idx] = -1.0
        combined_scores += penalty
        
        # Get top 5 recommendations
        top_k_indices = combined_scores.topk(5, dim=1).indices[0].tolist()
        
        # Add recommended items to the set for diversity penalty
        recommended_items.update(top_k_indices)
        
        # Display recommendations with images
        st.write(f"Top 5 item indices: {top_k_indices}")
        for idx in top_k_indices:
            item_info = df.iloc[idx]
            st.write(f"- Item Index: {idx}, Description: {item_info['description']}")
            
            # Display image if available
            image_paths = item_info.get('image_paths', [])
            if image_paths and isinstance(image_paths, list) and len(image_paths) > 0:
                image_path = image_paths[0]
                if os.path.exists(image_path):
                    st.image(image_path, caption=f"Image for Item {idx}", width=200)
                else:
                    st.write(f"Image for Item {idx} not found at path: {image_path}")
            else:
                st.write(f"No image available for Item {idx}")
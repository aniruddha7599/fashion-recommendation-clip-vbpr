👗 Fashion Recommendation System (CLIP + VBPR)
A personalized fashion recommender leveraging CLIP for multimodal embeddings and VBPR for ranking, developed as of July 16, 2025.

Personalized Rankings: Ranks items using VBPR based on user interaction history.
Interactive Demo: Streamlit app offers real-time, visual recommendations.
Diverse Suggestions: Applies a diversity penalty to prevent repetitive suggestions.
Category-Aware: Supports specific prompts (e.g., "men’s cotton shirt", "women’s leather handbag").
Efficient Preprocessing: Filters and aggregates data, ensuring quality inputs (items/users with ≥5 interactions).
Robust Evaluation: Computes Recall@5 and NDCG@5 across multiple categories.


📂 Project Structure
DL-Project/
├── data/ # ⚠ Local only (excluded from repo)
│   ├── preprocessed_fashion_data.pkl
│   ├── user_item_interactions.pkl
│   ├── meta_Clothing_Shoes_and_Jewelry.json # 8.2 GB
│   ├── Clothing_Shoes_and_Jewelry.json
│   └── images/ # Downloaded item images
├── models/ # ⚠ Local only (excluded from repo)
│   ├── clip_fashion_model/
│   │   ├── all_image_features.pt
│   │   └── vbpr_model.pt
├── preprocess.py # Data preprocessing script
├── main.py # CLIP fine-tuning and VBPR training
├── evaluate.py # Batch evaluation (Recall@5, NDCG@5)
├── streamlit_app.py # Streamlit interface
└── README.md # Project documentation


🔍 Dataset

Source: Amazon Fashion Dataset
Files: meta_Clothing_Shoes_and_Jewelry.json (metadata, ~8.2 GB), Clothing_Shoes_and_Jewelry.json (reviews)
Processing: Filters items with textual features (title, description, brand, categories) and image URLs, downloads one image per item, and creates .pkl files.


🛠️ Workflow

Preprocess (preprocess.py):
Loads and filters metadata/reviews, downloads high-resolution images, and verifies integrity.
Outputs preprocessed_fashion_data.pkl (item data), user_item_interactions.pkl (user-item pairs), and a ZIP file with images.


Train (main.py):
Fine-tunes CLIP (openai/clip-vit-base-patch32) with ITC Loss (15 epochs) and trains VBPR (10 epochs) using positive/negative item pairs.
Saves embeddings (all_image_features.pt) and model weights (vbpr_model.pt).


Evaluate (evaluate.py):
Assesses 100 random users across 9 categories, combining VBPR scores with CLIP text embeddings (weight=3.0) and a diversity penalty (-0.5).
Computes Recall@5 and NDCG@5.


Demo (streamlit_app.py):
Allows user/category selection, merges VBPR scores with category similarity (weight=75.0), and applies a diversity penalty (-1.0).
Displays top-5 items with images.




📊 Evaluation

Metrics: 
Recall@5: Proportion of relevant items in top-5.
NDCG@5: Ranking quality based on relevance.


Categories: shirt (men’s cotton shirt), pants (men’s denim pants), dress (women’s summer dress), shoes (men’s leather shoes), jacket (men’s leather jacket), necklace (women’s silver necklace), hat (men’s baseball hat), bag (women’s leather handbag), belt (men’s leather belt).


🚀 Getting Started
Prerequisites

Python 3.8+, PyTorch, Transformers, Streamlit, pandas, numpy, Pillow, tqdm, psutil, requests
Install: pip install torch transformers streamlit pandas numpy pillow tqdm psutil requests

Setup

Download Dataset: Place meta_Clothing_Shoes_and_Jewelry.json and Clothing_Shoes_and_Jewelry.json in data/.
Preprocess Data: python preprocess.py (creates .pkl files and ZIP).
Train Models: python main.py (fine-tunes CLIP, trains VBPR).
Evaluate: python evaluate.py (outputs metrics).
Run Demo: streamlit run streamlit_app.py (access at http://localhost:8501).


🖼️ Demo

Select a user and categories to see top-5 recommendations with images and descriptions.

📝 Notes

data/ and models/ are local-only due to size; use the ZIP file for data transfer.
Missing images are replaced with a default gray image (224x224).
Future enhancements: Fine-tune CLIP with fashion-specific data, optimize diversity penalty.


🤝 Contributing
Fork the repo, create a feature branch, commit changes, push, and open a PR.

📧 Contact
Contact me via GitHub.

📜 License
MIT License. See LICENSE for details.

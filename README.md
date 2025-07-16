👗 Fashion Recommendation System (CLIP + VBPR)
A personalized fashion recommender using CLIP for multimodal embeddings and VBPR for ranking, developed as of July 16, 2025.

Personalized Rankings: Ranks items using VBPR based on user interaction history.
Interactive Demo: Streamlit app offers real-time, visual recommendations.
Diverse Suggestions: Applies a diversity penalty to prevent repetition.
Category-Aware: Supports specific prompts (e.g., "men’s cotton shirt", "women’s leather handbag").
Efficient Preprocessing: Filters and aggregates data, ensuring quality inputs (items/users with ≥5 interactions).
Robust Evaluation: Computes Recall@5 and NDCG@5 across multiple categories.


📂 Project Structure

``` bash
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
```


🔍 Dataset

Source: Amazon Fashion Dataset (https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)
Files: meta_Clothing_Shoes_and_Jewelry.json (metadata, ~8.2 GB), Clothing_Shoes_and_Jewelry.json (reviews)
Processing: Filters items with textual features and image URLs, downloads one image per item, and creates .pkl files.


🛠️ Workflow

Preprocess (preprocess.py):
Loads and filters metadata/reviews, downloads high-resolution images, and verifies integrity.
Outputs preprocessed_fashion_data.pkl, user_item_interactions.pkl, and a ZIP file.


Train (main.py):
Fine-tunes CLIP (openai/clip-vit-base-patch32) with ITC Loss (15 epochs) and trains VBPR (10 epochs).
Saves embeddings and model weights.


Evaluate (evaluate.py):
Assesses 100 random users across 9 categories, combining VBPR scores with CLIP text embeddings and a diversity penalty.
Computes Recall@5 and NDCG@5.


Demo (streamlit_app.py):
Allows user/category selection, merges scores with category similarity, and displays top-5 items with images.




📊 Evaluation

Metrics: 
Recall@5: Proportion of relevant items in top-5.
NDCG@5: Ranking quality based on relevance.


Categories: shirt, pants, dress, shoes, jacket, necklace, hat, bag, belt


🚀 Getting Started
Prerequisites

Python 3.8+, PyTorch, Transformers, Streamlit, pandas, numpy, Pillow, tqdm, psutil, requests
Install: pip install torch transformers streamlit pandas numpy pillow tqdm psutil requests

Setup

Download Dataset: Place files in data/.
Preprocess: python preprocess.py (creates .pkl and ZIP).
Train: python main.py (fine-tunes CLIP, trains VBPR).
Evaluate: python evaluate.py (outputs metrics).
Run Demo: streamlit run streamlit_app.py (access at http://localhost:8501).


🖼️ Demo

![alt text](https://github.com/aniruddha7599/fashion-recommendation-clip-vbpr/blob/main/example.png)

Select a user and categories to see top recommendations with images.



🤝 Contributing
Fork, create a feature branch, commit, push, and open a PR.

📧 Contact
Contact me via GitHub.


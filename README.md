👗 Fashion Recommendation System (CLIP + VBPR)
A visual-aware personalized fashion recommender system powered by:

💡 CLIP (Contrastive Language–Image Pretraining) for extracting semantic image and text embeddings
🧠 VBPR (Visual Bayesian Personalized Ranking) for personalized user-item ranking

This system delivers multimodal recommendations using images and category prompts, featuring a Streamlit-based interactive demo and category-aware, diversity-penalized ranking. It processes the Amazon Fashion dataset to provide personalized fashion suggestions across categories like shirt, pants, shoes, and more.

📋 Features

Multimodal Recommendations: Combines CLIP's visual and textual embeddings for accurate suggestions.
Personalized Rankings: Uses VBPR to rank items based on user interaction history.
Interactive Demo: Streamlit app for real-time, user-specific recommendations with visual previews.
Diverse Suggestions: Applies a diversity penalty to avoid repetitive recommendations.
Category-Aware: Supports specific prompts (e.g., "men’s cotton shirt", "women’s leather handbag").
Efficient Preprocessing: Filters and aggregates data to ensure quality inputs (items/users with ≥5 interactions).
Robust Evaluation: Computes Recall@5 and NDCG@5 across multiple categories.


📂 Project Structure
DL-Project/
├── data/                        # ⚠ Local only (excluded from repo)
│   ├── preprocessed_fashion_data.pkl
│   ├── user_item_interactions.pkl
│   ├── meta_Clothing_Shoes_and_Jewelry.json  # 8.2 GB
│   ├── Clothing_Shoes_and_Jewelry.json
│   └── images/                  # Downloaded item images
├── models/                      # ⚠ Local only (excluded from repo)
│   ├── clip_fashion_model/
│   │   ├── all_image_features.pt
│   │   └── vbpr_model.pt
├── preprocess.py                # Data preprocessing script
├── main.py                      # CLIP fine-tuning and VBPR training
├── evaluate.py                  # Batch evaluation (Recall@5, NDCG@5)
├── streamlit_app.py             # Streamlit interface for live recommendations
└── README.md                    # Project documentation


🔍 Dataset
This project uses the Amazon Fashion dataset from UCSD.
📥 Download: Amazon Fashion Dataset
Required Files:

meta_Clothing_Shoes_and_Jewelry.json (metadata, ~8.2 GB)
Clothing_Shoes_and_Jewelry.json (user reviews)

Preprocessing:

Filters items with valid textual features (title, description, brand, categories) and image URLs.
Downloads one high-resolution image per item.
Filters users and items with ≥5 interactions.
Outputs preprocessed_fashion_data.pkl and user_item_interactions.pkl.


🛠️ Workflow
1. Data Preprocessing (preprocess.py)

Input: Raw Amazon Fashion dataset (metadata and reviews).
Steps:
Loads and filters metadata for items with textual features and image URLs.
Processes reviews to include user-item interactions with ratings (≥5 interactions).
Downloads one image per item, verifies integrity, and saves to data/images/.
Creates a ZIP file (amazon_fashion_data.zip) containing processed data and images.


Output:
preprocessed_fashion_data.pkl: Processed item data (textual features, image paths).
user_item_interactions.pkl: User-item interaction dictionary.



2. CLIP Fine-Tuning & VBPR Training (main.py)

CLIP Fine-Tuning:
Uses openai/clip-vit-base-patch32 to extract visual and textual embeddings.
Fine-tunes CLIP with ITC Loss (temperature=0.05) for 15 epochs.
Generates normalized image embeddings for all items.


VBPR Training:
Implements a matrix factorization model with visual embeddings (embedding_dim=100, visual_dim=512).
Trains for 10 epochs using MarginRankingLoss with positive/negative item pairs.


Output:
all_image_features.pt: Normalized CLIP image embeddings.
vbpr_model.pt: Trained VBPR model weights.
clip_fashion_model/: Fine-tuned CLIP model and processor.



3. Evaluation (evaluate.py)

Evaluates 100 random users across 9 categories: shirt, pants, dress, shoes, jacket, necklace, hat, bag, belt.
Combines VBPR scores with category-specific CLIP text embeddings (weight=3.0).
Applies a diversity penalty (-0.5) to avoid recommending the same items.
Computes:
Recall@5: Proportion of relevant items in top-5 recommendations.
NDCG@5: Ranking quality based on relevance.


Outputs per-category and overall average metrics.

4. Streamlit App (streamlit_app.py)

Interface:
Select a user ID and one or more categories (e.g., "men’s cotton shirt", "women’s summer dress").
Displays top-5 recommendations per category with item descriptions and images.


Scoring:
Combines VBPR scores (normalized to [-1, 1]) with category similarity scores (weight=75.0).
Applies a diversity penalty (-1.0) to ensure varied recommendations.


Debugging: Includes detailed logging for score normalization and image availability.


📊 Evaluation Metrics
Run evaluate.py to compute:

Recall@5: Measures the proportion of relevant items in the top-5 recommendations.
NDCG@5: Evaluates ranking quality based on relevance.

Supported Categories:

shirt (men’s cotton shirt)
pants (men’s denim pants)
dress (women’s summer dress)
shoes (men’s leather shoes)
jacket (men’s leather jacket)
necklace (women’s silver necklace)
hat (men’s baseball hat)
bag (women’s leather handbag)
belt (men’s leather belt)


🚀 Getting Started
Prerequisites

Python 3.8+
PyTorch
Transformers (openai/clip-vit-base-patch32)
Streamlit
Dependencies: pandas, numpy, Pillow, tqdm, psutil, requests
Install via:pip install torch transformers streamlit pandas numpy pillow tqdm psutil requests



Setup

Download the Dataset:

Get meta_Clothing_Shoes_and_Jewelry.json and Clothing_Shoes_and_Jewelry.json from the dataset link.
Place in the data/ directory.


Preprocess Data:
python preprocess.py


Outputs preprocessed_fashion_data.pkl, user_item_interactions.pkl, and images/ in a ZIP file.


Train Models:
python main.py


Fine-tunes CLIP, trains VBPR, and saves models/embeddings to models/clip_fashion_model/.


Evaluate Performance:
python evaluate.py


Outputs Recall@5 and NDCG@5 for each category and overall averages.


Run the Streamlit App:
streamlit run streamlit_app.py


Access the interactive demo at http://localhost:8501.




🖼️ Demo
Launch the Streamlit app to explore personalized fashion recommendations:

Select a user ID from the dropdown.
Choose one or more categories (e.g., shirt, shoes).
View top-5 recommendations per category with images and descriptions.
Diversity penalties ensure varied suggestions across categories.


📝 Notes

The data/ and models/ directories are excluded from the repository due to their size. Download and preprocess the dataset locally.
The preprocessing script creates a ZIP file (amazon_fashion_data.zip) for easy data transfer.
Missing images are handled with a default gray image (224x224) during training.
The Streamlit app includes debug logs for score normalization and image availability.
Future improvements:
Fine-tune CLIP with domain-specific fashion data.
Incorporate additional user features (e.g., preferences, demographics).
Optimize diversity penalty for better recommendation variety.




🤝 Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch (git checkout -b feature/YourFeature).
Commit changes (git commit -m 'Add YourFeature').
Push to the branch (git push origin feature/YourFeature).
Open a pull request.


📧 Contact
For questions or suggestions, open an issue or contact me via GitHub.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

\# 👗 Fashion Recommendation System (CLIP + VBPR)



This project implements a \*\*visual-aware personalized fashion recommender system\*\* using:



\- 💡 \*\*CLIP\*\* (Contrastive Language–Image Pretraining) for extracting \*\*semantic image and text embeddings\*\*

\- 🧠 \*\*VBPR\*\* (Visual Bayesian Personalized Ranking) for personalized user-item ranking



The system supports:

\- Multimodal recommendations (images + category prompts)

\- Streamlit-based interactive demo

\- Category-aware \& diversity-penalized ranking



---



\## 📂 Project Structure



DL Project/

├── data/ ← (⚠ Local only - excluded from repo)

│ ├── preprocessed\_fashion\_data.csv

│ ├── user\_item\_interactions.csv

│ ├── meta\_Clothing\_Shoes\_and\_Jewelry.csv (8.2 GB)

│ └── ...

├── models/ ← (⚠ Local only - excluded from repo)

│ ├── all\_image\_features.pt

│ └── vbpr\_model.pt

├── evaluate.py ← Batch evaluation script (Recall@5 / NDCG@5)

├── streamlit\_app.py ← Streamlit interface for live recommendation

├── main.ipynb ← Training \& feature extraction notebook

└── README.md 









---



\## 🔍 Dataset



This project uses the \*\*Amazon Fashion\*\* dataset from UCSD:



📥 \*\*\[Download the dataset here](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon\_v2/)\*\*  

➡ Recommended subset:  

\- `meta\_Clothing\_Shoes\_and\_Jewelry.json`

\- `reviews\_Clothing\_Shoes\_and\_Jewelry.json`



---



\## 🛠️ Features \& Workflow



\### 🔹 1. CLIP-Based Embeddings

\- Uses `openai/clip-vit-base-patch32` to extract:

&nbsp; - Visual embeddings of item images

&nbsp; - Text embeddings of category prompts



\### 🔹 2. VBPR (PyTorch)

\- A matrix factorization model that incorporates image features

\- Trained using user-item interaction data



\### 🔹 3. Streamlit App

\- Choose a user ID and fashion categories (e.g. "shirt", "shoes", "hat")

\- Visual recommendations ranked by user preference + semantic similarity

\- Diversity penalty ensures varied suggestions



---



\## 📊 Evaluation



Run `evaluate.py` to calculate:

\- \*\*Recall@5\*\*

\- \*\*NDCG@5\*\*



across categories like: `"shirt", "pants", "shoes", ...`







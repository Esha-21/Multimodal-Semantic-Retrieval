# Multimodal Image–Text Semantic Retrieval System (Zero-Shot)

## 📌 Project Overview
This project implements a **Multimodal Semantic Retrieval System** that supports **Text and Image** inputs for semantic search.

Using a pretrained **CLIP (Contrastive Language–Image Pretraining)** model, the system converts both text and images into a shared embedding space and retrieves the most semantically relevant results using **cosine similarity**.

The system works in a **zero-shot setting**, meaning no additional training or fine-tuning is required.

---

## 🎯 Key Features

### 🔹 Input Modalities
- 📝 **Text Input** – Natural language queries
- 🖼️ **Image Input** – Image-based similarity search

### 🔹 Hybrid Queries
- Text-only search
- Image-only search
- **Text + Image combined query**

### 🔹 Retrieval Capabilities
- Text → Image retrieval
- Image → Image similarity search
- Image → Text matching
- Zero-shot semantic search
- Top-K ranked results

### 🔹 System Features
- Cosine similarity–based matching
- Automatic modality detection
- Explainable similarity scores
- Confidence-aware retrieval

---

## 🏷️ Domain
- Multimodal Artificial Intelligence  
- Computer Vision  
- Natural Language Processing (NLP)  
- Information Retrieval  
- Semantic Search Systems  

---

## 🧠 How the System Works

1. **Text Encoding**  
   - User text queries are encoded using CLIP’s text encoder.

2. **Image Encoding**  
   - Images are encoded using CLIP’s image encoder.

3. **Shared Embedding Space**  
   - Both text and image embeddings lie in the same vector space.

4. **Similarity Matching**  
   - Cosine similarity is used to compare embeddings.
   - The system retrieves the **Top-K most semantically similar results**.

---

## 🔍 Supported Search Types

| Query Type | Supported |
|-----------|----------|
| Text → Image | ✅ |
| Image → Image | ✅ |
| Image → Text | ✅ |
| Text + Image (Hybrid) | ✅ |
| Zero-shot Inference | ✅ |

---

## ⚠️ Limitations
- Performs best on **general real-world images**
- Limited performance on:
  - Medical images (X-rays, MRI)
  - Satellite imagery
  - Technical diagrams
- Accuracy depends on image quality and clarity

---

## 🛠️ Tech Stack
- **Programming Language:** Python  
- **Model:** CLIP (Pretrained)  
- **Libraries:** PyTorch, NumPy  
- **Similarity Metric:** Cosine Similarity  
- **Frontend (Optional):** Streamlit / Flask  
- **Vector Search (Optional):** FAISS  

---

## 📂 Project Structure
Multimodal-Semantic-Retrieval/
│
├── data/
│ └── images/
├── embeddings/
│ └── image_embeddings.npy
├── clip_model.py
├── search.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

---

## ▶️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Esha-21/Multimodal-Semantic-Retrieval.git
cd Multimodal-Semantic-Retrieval

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the application
streamlit run app.py
"# -Multimodal-Semantic-Retrieval"  git init
"# -Multimodal-Semantic-Retrieval" 

# import streamlit as st
# from src.encode import encode_image, encode_text
# import torch.nn.functional as F
# from PIL import Image

# st.title("Multimodal CLIP AI System")

# image_file = st.file_uploader("Upload an image")
# text = st.text_input("Enter text")

# if image_file and text:
#     image = Image.open(image_file)
#     st.image(image)

#     img_emb = encode_image(image_file)
#     txt_emb = encode_text(text)

#     score = F.cosine_similarity(img_emb, txt_emb)
#     st.write("Similarity Score:", score.item())
import streamlit as st
import torch.nn.functional as F
from PIL import Image
import tempfile

from src.encode import encode_image, encode_text
from src.top_k_search import top_k_image_search

st.title("Multimodal CLIP AI System")

mode = st.radio(
    "Choose Mode",
    ["Image-Text Similarity", "Text → Image Search (Top-K)"]
)

# -------------------------------
# MODE 1: Image ↔ Text Similarity
# -------------------------------
if mode == "Image-Text Similarity":
    image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    text = st.text_input("Enter text")

    if image_file and text:
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_file.getbuffer())
            img_path = tmp.name

        img_emb = encode_image(img_path)
        txt_emb = encode_text(text)

        score = F.cosine_similarity(img_emb, txt_emb)

        image = Image.open(img_path)
        st.image(image)
        st.write("Cosine Similarity:", round(score.item(), 3))

# -------------------------------
# MODE 2: Text → Image Search
# -------------------------------
if mode == "Text → Image Search (Top-K)":
    text = st.text_input("Enter search text")
    k = st.slider("Top-K Results", 1, 5, 3)

    if text:
        results = top_k_image_search(text, k=k)

        for img, score in results:
            st.image(
                f"data/images/{img}",
                caption=f"{img} | Cosine Similarity: {score:.3f}"
            )

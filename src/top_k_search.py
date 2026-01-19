import os
import torch
import torch.nn.functional as F
from src.encode import encode_image, encode_text

def top_k_image_search(text, image_dir="data/images", k=3):
    # Encode text
    text_emb = encode_text(text)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    scores = []

    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)

        # Encode image
        img_emb = encode_image(img_path)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        # Cosine similarity
        score = F.cosine_similarity(text_emb, img_emb).item()
        scores.append((img_name, score))

    # Sort by similarity
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[:k]

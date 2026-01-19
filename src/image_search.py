import os
import torch.nn.functional as F
from src.encode import encode_image, encode_text

image_dir = "data/images"
query = encode_text("a dog running")

best_score = -1
best_image = None

for img in os.listdir(image_dir):
    emb = encode_image(f"{image_dir}/{img}")
    score = F.cosine_similarity(query, emb).item()
    if score > best_score:
        best_score = score
        best_image = img

print("Best Match:", best_image)

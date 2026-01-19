# import torch.nn.functional as F
# from src.encode import encode_image, encode_text

# img_emb = encode_image("data/images/sample.jpg")
# txt_emb = encode_text("a dog playing outside")

# score = F.cosine_similarity(img_emb, txt_emb)
# print("Similarity Score:", score.item())
import sys
import torch.nn.functional as F
from src.encode import encode_image, encode_text

def compute_similarity(image_path: str, text: str) -> float:
    img_emb = encode_image(image_path)
    txt_emb = encode_text(text)

    # cosine similarity (already normalized, but safe)
    score = F.cosine_similarity(img_emb, txt_emb, dim=-1)
    return score.item()


if __name__ == "__main__":
    # Usage:
    # python similarity.py "data/images/sample.jpg" "a dog playing outside"

    if len(sys.argv) < 3:
        print("Usage: python similarity.py <image_path> <text>")
        sys.exit(1)

    image_path = sys.argv[1]
    text = " ".join(sys.argv[2:])  # allows multi-word text

    similarity = compute_similarity(image_path, text)
    print(f"Cosine Similarity: {similarity:.4f}")

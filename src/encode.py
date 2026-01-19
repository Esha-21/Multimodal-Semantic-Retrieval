from PIL import Image
import torch
import clip
from src.load_clip import model, preprocess, device

# def encode_image(path):
#     image = preprocess(Image.open(path)).unsqueeze(0).to(device)
#     with torch.no_grad():
#         return model.encode_image(image)

# def encode_text(text):
#     tokens = clip.tokenize([text]).to(device)
#     with torch.no_grad():
#         return model.encode_text(tokens)
def encode_image(path):
    image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)  
    return image_features

def encode_text(text):
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)  
    return text_features
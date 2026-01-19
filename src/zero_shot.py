import clip, torch
from src.encode import encode_image
from src.load_clip import model, device

labels = ["a dog", "a cat", "a car", "a person"]
text_tokens = clip.tokenize(labels).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    image_features = encode_image("data/images/sample.jpg")
    probs = (image_features @ text_features.T).softmax(dim=-1)

for label, p in zip(labels, probs[0]):
    print(label, round(p.item(), 3))

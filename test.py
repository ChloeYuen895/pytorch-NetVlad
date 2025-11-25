# test.py
from PIL import Image
from torchvision import transforms
import torch, numpy as np
from models.NetVLAD import EmbedNet

model = EmbedNet('vgg16', num_clusters=64)
model.load_state_dict(torch.load('checkpoints/best_model.pth')['model_state_dict'])
model = model.cuda().eval()

transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

classes = ['arctic', 'forest', 'bamboo', 'grassland', 'desert']

def predict(path):
    img = Image.open(path).convert('RGB')
    vec = model(transform(img).unsqueeze(0).cuda()).cpu().numpy()[0]
    centroids = np.load('checkpoints/centroids.npy')
    pred = classes[np.argmin(np.linalg.norm(centroids - vec, axis=1))]
    print(f"→ {path}\n   PREDICTION: {pred.upper()}")

predict(r"C:\Users\yueny\Pictures\your_photo.jpg")
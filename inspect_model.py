import numpy as np
from PIL import Image
import cv2
import torch

k = 2

imagenet_labels = dict(enumerate(open("data/classes.txt")))

model = torch.load("data/model.pth")
model.eval()

img = (np.array(Image.open("data/dog.jpg")) / 128) - 1  # in the range -1, 1
img = cv2.resize(img, (384, 384))
inp = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(torch.float32)

# inspect model Patch Embedding layer output
B = inp.shape[0]
print('Model input shape: ', end='')
print(inp.shape)
print("Patch Embeddings output should be (batch_num, n_patches**2, embed_dim)")
print(f"n_patches = {384 / 16}")
print(f"n_patches**2 = {(384 / 16)**2}")
print('Inspecting model patchembed layer output')
patch_embed_out = model.patch_embed(inp)
print(patch_embed_out.shape)

# inspect model transformer layer output
output = model(inp)
"""
Train ViT model on Beans dataset using Hugging Face
Save the pytorch weights to a file
and load the weights into the model from scratch
"""

from datasets import load_dataset
import random
from PIL import ImageDraw, ImageFont, Image
import matplotlib.pyplot as plt
ds = load_dataset('beans')




from transformers import ViTFeatureExtractor

model_name_or_path = 'google/vit-base-patch16-384'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

print(feature_extractor)
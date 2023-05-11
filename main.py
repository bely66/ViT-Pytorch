import argparse

import numpy as np
import timm 
import torch

from modules import VisionTransformer

# Arguments parser (number of classes, output path)
parser = argparse.ArgumentParser()
parser.add_argument("--num_classes", type=int, default=3)
parser.add_argument("--output_path", type=str, default="data/model.pth")

args = parser.parse_args()

num_classes = args.num_classes
output_path = args.output_path


# Helpers
def get_n_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def assert_tensors_equal(t1, t2):
    a1, a2 = t1.detach().numpy(), t2.detach().numpy()

    np.testing.assert_allclose(a1, a2)


print("Loading official weights...")
model_name = "vit_base_patch16_384"
model_official = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
model_official.eval()
print(type(model_official))

custom_config = {
        "img_size": 384,
        "in_chans": 3,
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "qkv_bias": True,
        "mlp_ratio": 4,
        "num_classes": num_classes,
}

print("Loading custom weights...")
model_custom = VisionTransformer(**custom_config)
model_custom.eval()

print("Testing custom model vs official model -> Parameters .....")
for (n_o, p_o), (n_c, p_c) in zip(
        model_official.named_parameters(), model_custom.named_parameters()
):
    assert p_o.numel() == p_c.numel()
    print(f"{n_o} | {n_c}")

    p_c.data[:] = p_o.data

    assert_tensors_equal(p_c.data, p_o.data)

print("Testing custom model vs official model -> Forward pass .....")
# create random torch input
inp = torch.randn(1, 3, 384, 384)
res_c = model_custom(inp)
res_o = model_official(inp)
print("Custom model output shape: ", end="")
# Asserts
print("Asserting parameters are equal...")
assert get_n_params(model_custom) == get_n_params(model_official)
print("Asserting forward pass is equal...")
assert_tensors_equal(res_c, res_o)
# Save custom model
print("Saving custom model to: ", output_path)
torch.save(model_custom, output_path)

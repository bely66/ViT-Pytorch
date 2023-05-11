import torch 

def load_vit_model(model_path):
    print("Loading model from %s" % model_path)
    model = torch.load(model_path)
    return model

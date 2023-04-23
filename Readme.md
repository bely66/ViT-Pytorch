
# ViT using Pytorch

This repo is me trying to explore the Vision Transformer (ViT) model from google, using Pytorch.

The original paper can be found [here](https://arxiv.org/abs/2010.11929).

## Usage

## Installing Dependencies

- Having a python>=3.8 environment is recommended
- To install the dependencies, run:
`pip install -r requirements.txt`

## Exploring the code

- **modules.py:** Contains the implementation of the different modules/sub-networks used in the ViT model.
- **main.py:** Contains the code to test the model against the official implementation using timm module
- **test.py:** Contains the code to test the model with a sample image using the Coco weights, after running main.py
- **inspect.py:** Contains the code to inspect the model, layer by layer, after running main.py

## Running the code

1. Run `python main.py` to load the official model to the one we built. This will save the model weights in the `./data` folder.

2. Run `python test.py` to test the model with a sample image. This will print the top 5 predictions, I'm using a cat image feel free to change it.

3. Run `python inspect.py` to inspect the model, layer by layer. This will print the output of each layer.



# MyOwnLLaMA2: A a naive implementation of the LLaMA2 inference

MyOwnLLaMA2 is a text generation model built on PyTorch. This repository contains the implementation of the MyOwnLLaMA2 model, a class that encapsulates a transformer model and its tokenizer, and provides text generation functionalities. This model is capable of completing given text prompts using a transformer model. The implementation demonstrates how to build, load, and run a transformer model for text generation.

## Repository Contents

- `model.py` - Contains the implementation of the Transformer model, with Encoder Blocks, Self Attention and Feed Forward Networks along with necessary utility functions and classes for the model operation.
- `inference.py` - Contains the `MyOwnLLaMA2` class which encapsulates the transformer model and its tokenizer, and provides text generation functionalities, including a method for text completion.
- `requirements.txt` - Lists the dependencies required to run the model.

## Getting Started

### Prerequisites

Ensure you have the following dependencies installed. You can install them using the `requirements.txt` file:

- torch
- torchvision
- torchaudio
- sentencepiece
- tqdm

```bash
pip install -r requirements.txt
```

### Usage

1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Run the `inference.py` script to interact with the `MyOwnLLaMA2` model.

```bash
python inference.py
```

Upon running `inference.py`, the script initializes a `MyOwnLLaMA2` instance with the transformer model, tokenizer, and model arguments. It then demonstrates how to complete a given text prompt using the model.

## Model Architecture

The core model architecture is defined in `model.py`, which implements a transformer model with a specified number of encoder blocks, self attention mechanism, and feed forward networks. The transformer model is encapsulated by the `MyOwnLLaMA2` class in `inference.py`, which also provides a method for text completion.

## Authors

- Pavel Nakaznenko, 2023

## License

This project is licensed under the MIT License.

For more details on the model architecture and implementation, refer to the comments within `model.py` and `inference.py`.
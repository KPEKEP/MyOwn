# Captcha Recognition using PyTorch Lightning

This project demonstrates a Captcha recognition system using Convolutional Neural Networks (CNN) built with PyTorch and PyTorch Lightning. The model is trained on a custom dataset of captcha images to learn and predict the characters in unseen captcha images.

## Author

Pavel Nakaznenko, 2023

## Dataset

The dataset should be placed in a directory specified in the `config.yaml` file. Each image file name should contain the captcha text, separated by an underscore from other text in the filename.

Example:
01abc_1633028360.jpg

## Configuration

Configuration settings for the project are stored in `config.yaml` file, including:

- `folder_str`: Directory containing the captcha images.
- `target_height`: Height to resize images.
- `target_width`: Width to resize images.
- `char_num`: Number of characters in each captcha.
- `batch_size`: Batch size for training.
- `learning_rate`: Learning rate for optimizer.
- `patience_epoch_fraction`: Fraction of total batches per epoch before reducing learning rate upon plateau.
- `max_epochs`: Maximum number of training epochs.
- `alphabet`: Characters to be recognized in captchas.
- `img_format`: Image file format.

## Usage

1. Ensure that the `captchanet.py`, `config.yaml`, and the dataset directory are in the project root.
2. Install the required libraries listed in `requirements.txt` using pip:
bash
pip install -r requirements.txt

3. Run the main script to start training the model:
bash
python captchanet.py

The script will train the model on the dataset, saving model checkpoints, and logging training/validation loss to TensorBoard. The model checkpoint with the lowest validation loss will be saved in the `checkpoints` directory.

## License

This project is open source under the [MIT license](LICENSE).
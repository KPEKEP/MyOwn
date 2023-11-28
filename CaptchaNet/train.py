# CNN for simple alphanumeric Captcha recognition
# Pavel Nakaznenko, 2023

import argparse
import yaml
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from captchanet import CaptchaNet, CaptchasDataset, InpatientScheduler


def load_config(file_path):
    """Loads configuration from a YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def create_datasets(config):
    """Creates train and validation datasets based on the configuration."""
    dataset = CaptchasDataset(
        config['folder_str'],
        (config['target_height'], config['target_width']),
        config['alphabet'],
        config['img_format']
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    return random_split(dataset, [train_size, val_size])


def create_data_loaders(train_dataset, val_dataset, batch_size):
    """Creates data loaders for training and validation."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def create_model(config, train_loader):
    """Creates the CaptchaNet model based on the configuration."""
    return CaptchaNet(
        config['target_height'],
        config['target_width'],
        config['char_num'],
        len(config['alphabet']),
        len(train_loader),
        config['learning_rate'],
        config['patience_epoch_fraction']
    )


def create_checkpoint_callback():
    """Creates a ModelCheckpoint callback for saving model checkpoints."""
    return ModelCheckpoint(
        dirpath='checkpoints',
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=-1,
    )


def train_model(model, train_loader, val_loader, max_epochs):
    """Trains the model using PyTorch Lightning."""
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=pl.loggers.WandbLogger(),
        callbacks=[create_checkpoint_callback()]
    )
    trainer.fit(model, train_loader, val_loader)


def main(args):
    config = load_config(args.config)

    train_dataset, val_dataset = create_datasets(config)
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, config['batch_size'])

    model = create_model(config, train_loader)

    train_model(model, train_loader, val_loader, config['max_epochs'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file.')
    args = parser.parse_args()
    main(args)

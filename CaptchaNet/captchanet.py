# CNN for simple alphanumeric Captcha recognition
# Pavel Nakaznenko, 2023

import glob
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from pytorch_lightning import LightningModule


class CaptchasDataset(Dataset):
    """Dataset class for loading captcha images and labels."""
    
    def __init__(self, image_dir, resize_dims, alphabet, img_format, transform=None):
        self.image_paths = glob.glob(os.path.join(image_dir, f'*.{img_format}'))
        self.resize_dims = resize_dims
        self.alphabet = alphabet
        self.transform = transform or self.default_transform()

    @staticmethod
    def default_transform():
        """Default transformation pipeline for images."""
        return transforms.Compose([
            transforms.Resize(resize_dims),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Grayscale(),            
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image_tensor = self.transform(image)
        image_name = os.path.basename(image_path).split('.')[0].split("_")[0]
        label_np = np.array([self.alphabet.index(char) for char in image_name])
        label = torch.tensor(label_np, dtype=torch.long)
        return image_tensor, label


class InpatientScheduler:
    """Custom learning rate scheduler with patience."""
    
    def __init__(self, optimizer, total_batches_in_epoch, patience_fraction, factor=0.5):
        self.optimizer = optimizer
        self.patience_fraction = patience_fraction
        self.factor = factor
        self.best_loss = float('inf')
        self.bad_batches = 0
        self.total_batches_in_epoch = total_batches_in_epoch

    def step(self, val_loss, batch_idx):
        if self.total_batches_in_epoch is None:
            raise ValueError("total_batches_in_epoch must be set")
        
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.bad_batches = 0
        else:
            self.bad_batches += 1
        
        patience_batches = self.patience_fraction * self.total_batches_in_epoch
        if self.bad_batches >= patience_batches:
            self.bad_batches = 0
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.factor
            logging.warning(f"Patience is over! Adjusted the learning rate")


class CaptchaNet(LightningModule):
    """Neural network model for captcha recognition."""
    
    def __init__(self, img_height, img_width, num_of_letters, num_of_variations,
                 total_batches_in_epoch=1, lr=0.001, patience_epoch_fraction=1):
        super(CaptchaNet, self).__init__()
        
        self.img_height = img_height
        self.img_width = img_width
        self.num_of_letters = num_of_letters
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(5, 5), padding=2)
        self.bn2 = nn.BatchNorm2d(48)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(5, 5), padding=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.dropout1 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(64 * (img_height // 8) * (img_width // 8), 512)
        self.dropout2 = nn.Dropout(0.1)
        
        self.character_layers = nn.ModuleList([nn.Linear(512, num_of_variations) for _ in range(num_of_letters)])
        
        self.total_batches_in_epoch = total_batches_in_epoch
        self.patience_epoch_fraction = patience_epoch_fraction
        self.lr = lr

    def forward(self, x):
        x = self.layer_block(x, self.conv1, self.bn1, self.pool1)
        x = self.layer_block(x, self.conv2, self.bn2, self.pool2)
        x = self.layer_block(x, self.conv3, self.bn3, self.pool3)
        
        x = self.dropout1(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        
        outputs = [char_layer(x) for char_layer in self.character_layers]
        return torch.stack(outputs, dim=1)
    
    @staticmethod
    def layer_block(x, conv, bn, pool):
        """Block of layers including Convolution, BatchNorm, ReLU, and MaxPooling."""
        x = F.relu(bn(conv(x)))
        x = pool(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = sum(F.cross_entropy(outputs[:, i, :], labels[:, i]) for i in range(self.num_of_letters))
        acc = sum((torch.argmax(outputs[:, i, :], dim=1) == labels[:, i]).float().mean() for i in range(self.num_of_letters)) / self.num_of_letters
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.custom_scheduler.step(loss, batch_idx)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = sum(F.cross_entropy(outputs[:, i, :], labels[:, i]) for i in range(self.num_of_letters))
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        self.custom_scheduler = InpatientScheduler(optimizer, self.total_batches_in_epoch, patience_fraction=self.patience_epoch_fraction)
        return optimizer
    
    def recognize_captcha(self, image_path, alphabet):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess_image(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            outputs = self.forward(image_tensor.to(self.device))
            predictions = []

            for i in range(outputs.size(1)):
                predicted_idx = torch.argmax(outputs[:, i, :], dim=1)
                predictions.append(alphabet[predicted_idx]) 

            recognized_text = ''.join(predictions)
        return recognized_text

    def preprocess_image(self, image):
        """Preprocesses the image for inference."""
        transform = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Grayscale(),         
        ])
        return transform(image)
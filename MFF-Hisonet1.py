import cv2
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from keras.utils.np_utils import to_categorical
from src.models.Model_BackboneNCA import BackboneNCA
from src.losses.LossFunctions import HybridLoss

# Dataset Loader with CLAHE enhancement
def Dataset_loader(DIR, RESIZE, use_clahe=True):
    IMG = []
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    for IMAGE_NAME in tqdm(os.listdir(DIR)):
        PATH = os.path.join(DIR, IMAGE_NAME)
        _, ftype = os.path.splitext(PATH)
        if ftype == ".png":  # Adjust based on image type
            img = read(PATH)
            img = cv2.resize(img, (RESIZE, RESIZE), interpolation=cv2.INTER_LINEAR)  # Bilinear interpolation
            if use_clahe:
                img = apply_clahe(img)
            IMG.append(np.array(img))
    return IMG

# CLAHE function to enhance contrast of the images
def apply_clahe(img):
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_clahe = cv2.merge((l, a, b))
    return cv2.cvtColor(img_clahe, cv2.COLOR_LAB2RGB)

# Load the dataset
benign_train = np.array(Dataset_loader('/content/drive/MyDrive/Inbreast_dataset/data/Train/benign', 256))
malign_train = np.array(Dataset_loader('/content/drive/MyDrive/Inbreast_dataset/data/Train/maligant', 256))
benign_test = np.array(Dataset_loader('/content/drive/MyDrive/Inbreast_dataset/data/Validation/benign', 256))
malign_test = np.array(Dataset_loader('/content/drive/MyDrive/Inbreast_dataset/data/Validation/maligant', 256))

# Create labels for benign (0) and malignant (1) data
benign_train_label = np.zeros(len(benign_train))
malign_train_label = np.ones(len(malign_train))
benign_test_label = np.zeros(len(benign_test))
malign_test_label = np.ones(len(malign_test))

# Merge data and labels
X_train = np.concatenate((benign_train, malign_train), axis=0)
Y_train = np.concatenate((benign_train_label, malign_train_label), axis=0)
X_test = np.concatenate((benign_test, malign_test), axis=0)
Y_test = np.concatenate((benign_test_label, malign_test_label), axis=0)

# Shuffle the training data
s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
Y_train = Y_train[s]

# Shuffle the test data
s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test = X_test[s]
Y_test = Y_test[s]

# Convert labels to categorical (binary classification)
Y_train = to_categorical(Y_train, num_classes=2)
Y_test = to_categorical(Y_test, num_classes=2)

# Global Feature Enhancement Module (GFEM)
class GFEM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GFEM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        global_feat = self.pool(x)
        global_feat = torch.flatten(global_feat, 1)
        global_feat = torch.tanh(self.fc(global_feat))
        return global_feat

# Local Feature Enhancement Module (LFEM)
class LFEM(nn.Module):
    def __init__(self, in_channels, out_channels, k=4):
        super(LFEM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        local_feat = self.pool(x)
        return local_feat

# Dual Context Aggregation Module
class DualContextAggregation(nn.Module):
    def __init__(self, gfem_out, lfem_out, out_channels):
        super(DualContextAggregation, self).__init__()
        self.gfem_out = gfem_out
        self.lfem_out = lfem_out
        self.alpha = nn.Parameter(torch.ones(1))  # Parameter for combining GFEM and LFEM
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, gfem_feat, lfem_feat):
        return self.alpha * gfem_feat + self.beta * lfem_feat

# Bi-CBMSegNet Model with ResNet152 as Encoder
class BiCBMSegNet(nn.Module):
    def __init__(self):
        super(BiCBMSegNet, self).__init__()
        self.encoder = BackboneNCA(input_channels=3, output_channels=32, pretrained=True)  # Adjusted for RGB input
        self.gfem = GFEM(in_channels=32, out_channels=64)
        self.lfem = LFEM(in_channels=32, out_channels=64)
        self.agg = DualContextAggregation(gfem_out=64, lfem_out=64, out_channels=64)
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        x = self.encoder(x)
        gfem_feat = self.gfem(x)
        lfem_feat = self.lfem(x)
        agg_feat = self.agg(gfem_feat, lfem_feat)
        output = self.decoder(agg_feat)
        return output

# Hybrid Loss Function (BCE + Dice)
def hybrid_loss(predictions, targets, alpha, gamma):
    bce_loss = nn.BCEWithLogitsLoss()(predictions, targets)
    
    predictions = torch.sigmoid(predictions)
    intersection = (predictions * targets).sum()
    dice_loss = 1 - (2. * intersection + 1e-5) / (predictions.sum() + targets.sum() + 1e-5)

    return gamma * dice_loss + (1 - gamma) * bce_loss

# Training function
def train_model(model, X_train, Y_train, optimizer, alpha, gamma, epochs):
    model.train()
    for epoch in range(epochs):
        for batch_idx in range(0, len(X_train), config['batch_size']):
            data = torch.tensor(X_train[batch_idx:batch_idx+config['batch_size']]).float().to(config['device'])
            target = torch.tensor(Y_train[batch_idx:batch_idx+config['batch_size']]).float().to(config['device'])
            
            optimizer.zero_grad()
            output = model(data)
            loss = hybrid_loss(output, target, alpha, gamma)
            loss.backward()
            optimizer.step()

            if batch_idx % config['step_per_epoch'] == 0:
                print(f'Epoch: {epoch+1}, Step: {batch_idx}, Loss: {loss.item()}')

        if (epoch + 1) % config['save_interval'] == 0:
            torch.save(model.state_dict(), config['model_path'] + f'_epoch_{epoch+1}.pth')

# Evaluation function
def evaluate_model(model, X_test, Y_test):
    model.eval()
    acc, dice = 0, 0
    with torch.no_grad():
        for batch_idx in range(0, len(X_test), config['batch_size']):
            data = torch.tensor(X_test[batch_idx:batch_idx+config['batch_size']]).float().to(config['device'])
            target = torch.tensor(Y_test[batch_idx:batch_idx+config['batch_size']]).float().to(config['device'])
            output = model(data)
            pred = torch.sigmoid(output)
            
            acc += ((pred > 0.5) == target).float().mean().item()
            intersection = (pred * target).sum()
            dice += (2. * intersection + 1e-5) / (pred.sum() + target.sum() + 1e-5)

    acc /= (len(X_test) // config['batch_size'])
    dice /= (len(X_test) // config['batch_size'])
    return {'Accuracy': acc, 'Dice Score': dice}

# Main function
def main():
    model = BiCBMSegNet().to(config['device'])
    optimizer = SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])
    scheduler = ExponentialLR(optimizer, gamma=0.99)

    # Train the model
    train_model(model, X_train, Y_train, optimizer, config['alpha'], config['gamma'], config['n_epoch'])

    # Evaluate the model
    evaluation_metrics = evaluate_model(model, X_test, Y_test)
    print(f"Evaluation Metrics: {evaluation_metrics}")

# Configuration
config = {
    'img_path': r"image_path",  # Path to the image dataset
    'label_path': r"label_path",  # Path to the segmentation masks
    'model_path': r'Models/Med_NCA_Run1',
    'device': "cuda:0",
    'lr': 1e-3,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'save_interval': 10,
    'evaluate_interval': 10,
    'n_epoch': 30,  # Set to 30 epochs
    'batch_size': 32,
    'step_per_epoch': 100,
    'input_size': (256, 256),  # Input size set to 256x256
    'alpha': 0.5,
    'gamma': 1.0,
}

if __name__ == "__main__":
    main()

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D
from src.models.BiCBMSegNet import BiCBMSegNet  # Assuming BiCBMSegNet is implemented in this file
from src.losses.LossFunctions import HybridLoss
from src.utils.Experiment import Experiment

# Define a custom agent class for Bi-CBMSegNet
class Agent_Mff-histoNet:
    def __init__(self, model, optimizer, scheduler, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(config['device'])
        self.config = config
        self.epoch = 0

    def train(self, data_loader, loss_function):
        self.model.train()
        for epoch in range(self.config['n_epoch']):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(data_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)

                # Compute loss
                loss = loss_function(outputs, labels)

                # Backward pass
                loss.backward()

                # Optimize
                self.optimizer.step()

                running_loss += loss.item()

                if i % self.config['save_interval'] == 0:  # Save model at intervals
                    print(f"Epoch [{epoch+1}/{self.config['n_epoch']}], Step [{i+1}], Loss: {loss.item():.4f}")

            # Step the scheduler
            self.scheduler.step()
            print(f'Epoch [{epoch+1}/{self.config['n_epoch']}], Average Loss: {running_loss / len(data_loader):.4f}')

    def evaluate(self, data_loader, loss_function):
        self.model.eval()
        total_dice_score = 0.0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)

                # Compute Dice score or any other evaluation metrics
                dice_score = self.compute_dice_score(outputs, labels)
                total_dice_score += dice_score

        average_dice_score = total_dice_score / len(data_loader)
        return average_dice_score

    def compute_dice_score(self, outputs, labels):
        smooth = 1e-5
        outputs = torch.sigmoid(outputs)
        intersection = (outputs * labels).sum()
        dice = (2. * intersection + smooth) / (outputs.sum() + labels.sum() + smooth)
        return dice.item()

# Configuration for Bi-CBMSegNet
config = [{
    # Basic
    'img_path': r"image_path",  # Path to mammogram images
    'label_path': r"label_path",  # Path to segmentation masks
    'model_path': r'Models/BiCBMSegNet_Run1',
    'device': "cuda:0",
    'unlock_CPU': True,
    # Optimizer
    'lr': 1e-3,
    'lr_gamma': 0.9999,
    'betas': (0.9, 0.999),  # Standard Adam betas
    # Training
    'save_interval': 10,
    'evaluate_interval': 10,
    'n_epoch': 30,  # Set to 30 epochs based on Bi-CBMSegNet experiments
    'batch_size': 32,
    # Model
    'channel_n': 32,  # Number of channels
    'inference_steps': 64,
    'cell_fire_rate': 0.5,
    'input_channels': 3,  # RGB Input
    'output_channels': 1,  # Binary mask output
    'hidden_size': 128,
    # Data
    'input_size': (256, 256),  # Resized to 256x256 as per Bi-CBMSegNet
    'data_split': [0.8, 0.2],  # Train-test split (80% train, 20% test)
}]

# Initialize dataset and model components
dataset = Dataset_NiiGz_3D(config[0]['img_path'], config[0]['label_path'], slice=2, transform=None)  # Assuming 2D mammogram slices
device = torch.device(config[0]['device'])
model = Mff-histoNet(config[0]['channel_n'], config[0]['input_channels'], config[0]['output_channels']).to(device)

# Optimizer and Scheduler
optimizer = SGD(model.parameters(), lr=config[0]['lr'], momentum=0.9, weight_decay=1e-4)
scheduler = ExponentialLR(optimizer, gamma=config[0]['lr_gamma'])

# Initialize Agent for Bi-CBMSegNet
agent = Agent_BiCBMSegNet(model, optimizer, scheduler, config[0])

# Set experiment and data loaders
exp = Experiment(config, dataset, model, agent)
dataset.set_experiment(exp)
data_loader = DataLoader(dataset, shuffle=True, batch_size=config[0]['batch_size'])

# Loss function: Hybrid Loss combining BCE and Dice
loss_function = HybridLoss(alpha=0.5, gamma=1.0)

# Train the model
agent.train(data_loader, loss_function)



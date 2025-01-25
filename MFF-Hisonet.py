# Import Required Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from skimage import io, color, exposure
from skimage.color import rgb2hed, hed2rgb
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import torchvision.models as models
from torchcam.methods import GradCAM
from tqdm import tqdm
from PIL import Image


class BreastCancerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = io.imread(image_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def extract_glcm_features(image):
    gray = color.rgb2gray(image)
    glcm = greycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    features = [
        greycoprops(glcm, 'contrast').ravel(),
        greycoprops(glcm, 'dissimilarity').ravel(),
        greycoprops(glcm, 'homogeneity').ravel(),
        greycoprops(glcm, 'energy').ravel(),
        greycoprops(glcm, 'correlation').ravel(),
        greycoprops(glcm, 'ASM').ravel()
    ]
    return np.concatenate(features)

def extract_lbp_features(image, radius=3, n_points=24):
    gray = color.rgb2gray(image)
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize
    return hist

def extract_gabor_features(image, frequencies=[0.1, 0.5, 0.9], theta=0):
    gray = color.rgb2gray(image)
    kernels = []
    for frequency in frequencies:
        kernel = cv2.getGaborKernel((21, 21), 5, theta, frequency, 0.5, 0, ktype=cv2.CV_32F)
        kernels.append(kernel)
    features = []
    for kernel in kernels:
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        features.append(filtered.mean())
        features.append(filtered.var())
    return np.array(features)

# Step 3: Define the DenseNet121 Base Model
class DenseNet121Base(nn.Module):
    def __init__(self, num_classes=2):
        super(DenseNet121Base, self).__init__()
        self.base_model = models.densenet121(pretrained=True)
        # Freeze all layers
        for param in self.base_model.parameters():
            param.requires_grad = False
        # Replace the classifier
        self.base_model.classifier = nn.Linear(self.base_model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)


class QTN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(QTN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.tensor_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.tensor_network(x)  # Tensor network operations
        x = self.fc2(x)
        return x


class MFFHistoNet(nn.Module):
    def __init__(self, cnn_model, qtn_model, texture_feature_dim, num_classes=2):
        super(MFFHistoNet, self).__init__()
        self.cnn_model = cnn_model
        self.qtn_model = qtn_model
        self.fc_texture = nn.Linear(texture_feature_dim, 128)  # Texture feature embedding
        self.fc_fusion = nn.Linear(cnn_model.base_model.classifier.out_features + qtn_model.fc2.out_features + 128, 256)  # Fusion layer
        self.fc_final = nn.Linear(256, num_classes)

    def forward(self, x):
        # CNN features
        cnn_features = self.cnn_model(x)

        # QTN features
        qtn_features = self.qtn_model(x.view(x.size(0), -1))

        # Texture features
        texture_features = []
        for img in x:
            img_np = img.permute(1, 2, 0).cpu().numpy()  # Convert to numpy for texture extraction
            glcm = extract_glcm_features(img_np)
            lbp = extract_lbp_features(img_np)
            gabor = extract_gabor_features(img_np)
            texture_features.append(np.concatenate([glcm, lbp, gabor]))
        texture_features = torch.tensor(np.array(texture_features), dtype=torch.float32).to(x.device)
        texture_features = self.fc_texture(texture_features)

        # Feature fusion
        combined_features = torch.cat((cnn_features, qtn_features, texture_features), dim=1)
        fused_features = torch.relu(self.fc_fusion(combined_features))  # Fusion layer
        output = self.fc_final(fused_features)
        return output


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

    return train_losses, val_losses, train_accuracies, val_accuracies


def load_data():

    base_dir = "/content/drive/MyDrive/data"  # Update this path to your dataset location
    train_benign_dir = os.path.join(base_dir, "Train/benign")
    train_malignant_dir = os.path.join(base_dir, "Train/maligant")
    test_benign_dir = os.path.join(base_dir, "Validation/benign")
    test_malignant_dir = os.path.join(base_dir, "Validation/maligant")


    def load_images_from_folder(folder):
        images = []
        for filename in tqdm(os.listdir(folder)):
            img_path = os.path.join(folder, filename)
            if img_path.endswith(".png"):
                img = cv2.imread(img_path)
                img = cv2.resize(img, (224, 224))
                images.append(img)
        return np.array(images)

    benign_train = load_images_from_folder(train_benign_dir)
    malign_train = load_images_from_folder(train_malignant_dir)
    benign_test = load_images_from_folder(test_benign_dir)
    malign_test = load_images_from_folder(test_malignant_dir)


    benign_train_label = np.zeros(len(benign_train))
    malign_train_label = np.ones(len(malign_train))
    benign_test_label = np.zeros(len(benign_test))
    malign_test_label = np.ones(len(malign_test))


    X_train = np.concatenate((benign_train, malign_train), axis=0)
    Y_train = np.concatenate((benign_train_label, malign_train_label), axis=0)
    X_test = np.concatenate((benign_test, malign_test), axis=0)
    Y_test = np.concatenate((benign_test_label, malign_test_label), axis=0)


    s = np.arange(X_train.shape[0])
    np.random.shuffle(s)
    X_train = X_train[s]
    Y_train = Y_train[s]

    s = np.arange(X_test.shape[0])
    np.random.shuffle(s)
    X_test = X_test[s]
    Y_test = Y_test[s]


    Y_train = np.eye(2)[Y_train.astype(int)]
    Y_test = np.eye(2)[Y_test.astype(int)]


    X_train = torch.tensor(X_train.transpose(0, 3, 1, 2), dtype=torch.float32)
    X_test = torch.tensor(X_test.transpose(0, 3, 1, 2), dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)


    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    val_dataset = torch.utils.data.TensorDataset(X_test, Y_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader


def main():

    train_loader, val_loader = load_data()


    cnn_model = DenseNet121Base(num_classes=2)  # Binary classification (benign vs malignant)
    qtn_model = QTN(input_dim=224*224*3, hidden_dim=512, output_dim=128)
    texture_feature_dim = 6 + 26 + 6  # GLCM (6) + LBP (26) + Gabor (6)
    mff_histonet = MFFHistoNet(cnn_model, qtn_model, texture_feature_dim, num_classes=2)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mff_histonet.parameters(), lr=0.001)


    train_losses, val_losses, train_accuracies, val_accuracies = train_model(mff_histonet, train_loader, val_loader, criterion, optimizer, num_epochs=10)


    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.show()


    evaluate_model(mff_histonet, val_loader)


    torch.save(mff_histonet.state_dict(), 'mff_histonet.pth')


def evaluate_model(model, val_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.argmax(dim=1).cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    main()
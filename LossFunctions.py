import torch
import torch.nn.functional as F

class DiceLoss(torch.nn.Module):
    def __init__(self, useSigmoid=True):
        self.useSigmoid = useSigmoid
        super(DiceLoss, self).__init__()

    def forward(self, input, target, smooth=1):
        if self.useSigmoid:
            input = torch.sigmoid(input)
        input = torch.flatten(input)
        target = torch.flatten(target)
        intersection = (input * target).sum()
        dice = (2. * intersection + smooth) / (input.sum() + target.sum() + smooth)
        return 1 - dice

class DiceBCELoss(torch.nn.Module):
    def __init__(self, useSigmoid=True):
        self.useSigmoid = useSigmoid
        super(DiceBCELoss, self).__init__()

    def forward(self, input, target, smooth=1):
        input = torch.sigmoid(input)
        input = torch.flatten(input)
        target = torch.flatten(target)
        intersection = (input * target).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (input.sum() + target.sum() + smooth)
        BCE = F.binary_cross_entropy(input, target, reduction='mean')
        return BCE + dice_loss

class BCELoss(torch.nn.Module):
    def __init__(self, useSigmoid=True):
        self.useSigmoid = useSigmoid
        super(BCELoss, self).__init__()

    def forward(self, input, target):
        input = torch.sigmoid(input)
        input = torch.flatten(input)
        target = torch.flatten(target)
        return F.binary_cross_entropy(input, target, reduction='mean')

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        input = torch.sigmoid(input)
        input = torch.flatten(input)
        target = torch.flatten(target)
        logit = input.clamp(self.eps, 1. - self.eps)
        loss_bce = F.binary_cross_entropy(input, target, reduction='mean')
        focal_loss = loss_bce * (1 - logit) ** self.gamma
        return focal_loss.mean()

class DiceFocalLoss(FocalLoss):
    def __init__(self, gamma=2, eps=1e-7):
        super(DiceFocalLoss, self).__init__(gamma, eps)

    def forward(self, input, target):
        input = torch.sigmoid(input)
        input = torch.flatten(input)
        target = torch.flatten(target)
        intersection = (input * target).sum()
        dice_loss = 1 - (2. * intersection + 1.) / (input.sum() + target.sum() + 1.)
        logit = input.clamp(self.eps, 1. - self.eps)
        loss_bce = F.binary_cross_entropy(input, target, reduction='mean')
        focal_loss = loss_bce * (1 - logit) ** self.gamma
        return focal_loss.mean() + dice_loss

model = BiCBMSegNet(in_channels=3, out_channels=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_function = DiceBCELoss()

for epoch in range(num_epochs):
    for data, labels in data_loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

import torch.nn as nn
import torch
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=2,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,kernel_size=2,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(100,256),
            nn.ReLU(),
            nn.Linear(256,128)
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class ConstructiveLearning(nn.Module):
    def __init__(self,temperature,device):
        super(ConstructiveLearning,self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self,image_org,image_aug):
        batch_size = image_org.size(0)
        flattened_image = torch.cat([image_aug,image_org],dim=0)

        similarity_matrix = torch.matmul(flattened_image,flattened_image.T)/self.temperature

        #since in a matrix the diagonal represents the maximmum relation between them we create mask matrix
        mask = torch.eye(2*batch_size).bool()

        #use either the positive mask value or mask value to create similarity embeddings
        positive_mask = torch.cat([torch.arange(batch_size),torch.arange(batch_size)+batch_size]).view(-1,1)
        similarity_matrix = similarity_matrix[~positive_mask].view(2*batch_size,-1).to(self.device)
        labels = torch.zeros(2*batch_size).to(self.device)
        loss = nn.CrossEntropyLoss()(similarity_matrix,labels.long())
        return loss


import torch.nn as nn
import torch
import torch.nn.functional as F
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
            nn.Linear(12800,256),
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
        batch_size = image_org.shape[0]
        flattened_image = torch.cat([image_aug,image_org],dim=0)

        similarity_matrix = torch.matmul(flattened_image,flattened_image.T)/self.temperature

        #since in a matrix the diagonal represents the maximmum relation between them we create mask matrix
        mask = torch.eye(2*batch_size).bool()

        #use either the positive mask value or mask value to create similarity embeddings
        positive_mask = torch.cat([torch.arange(batch_size),torch.arange(batch_size)+batch_size]).view(-1,1)
        similarity_matrix = similarity_matrix[~mask].view(2*batch_size,-1).to(self.device)
        labels = torch.zeros(2*batch_size).to(self.device)
        loss = nn.CrossEntropyLoss()(similarity_matrix,labels.long())
        return loss

class CLR_Pytorch(nn.Module):
    def __init__(self, temperature,device="cpu"):
        super(CLR_Pytorch,self).__init__()
        self.temperature = temperature
        self.device = device
    def forward(self,image):

        batch_size = image.shape[0]

        target = torch.arange(batch_size).to(self.device)
        target[0::2] += 1
        target[1::2] -= 1


        similarity_matrix = F.cosine_similarity(image[None,:,:],image[:,None,:],dim=1).to(self.device)

        #creating mask of positive pairs to avoid influencing the model
        eye = torch.eye(batch_size).bool()
        similarity_matrix[eye] = float('-inf') # avoiding similar pairs

        loss = F.cross_entropy(similarity_matrix/self.temperature,target,reduction='mean')
        return loss

class CLR_two_label(nn.Module):
    def __init__(self,temperature,device,positive_labels):
        super(CLR_two_label,self).__init__()
        self.temperature = temperature
        self.device = device
        self.positive = positive_labels

    def forward(self,image):

        batch_size = image.shape[0]
        similarity_matrix = F.cosine_similarity(image[None,:,:],image[:,None,:],dim=1).to(self.device)
        eye = torch.eye(batch_size,image.shape[1]).bool()
        similarity_matrix[eye] = float('-inf')

        pos_indices = self.positive
        target = torch.zeros(batch_size,image.shape[1]).to(self.device)
        target[pos_indices[:,0],pos_indices[:,1]] = 1

        loss = F.binary_cross_entropy((similarity_matrix/self.temperature).sigmoid(),target,reduction='none')
        target_positive = target.bool().to(self.device)
        target_negative = ~target_positive.to(self.device)

        loss_positive = torch.zeros(image.shape[0],image.shape[1]).to(self.device).masked_scatter(target_positive,
                                                                                                  loss[target_positive]).to(self.device)
        loss_negative = torch.zeros(image.shape[0],image.shape[1]).to(self.device).masked_scatter(target_negative,
                                                                                                  loss[target_negative]).to(self.device)

        #summing up batches loss values
        loss_positive = loss_positive.sum(dim=0)
        loss_negative = loss_negative.sum(dim=0)

        num_pos = target.sum(dim=0)
        num_neg = target.size(1) - num_pos

        return ((loss_positive / num_pos) + (loss_negative / num_neg))






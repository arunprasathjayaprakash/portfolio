import torch
import os
from model import ConstructiveLearning,CLR_two_label , SimpleCNN
from pre_processing import load_data
from tqdm import tqdm

def train_model(model,train_loader,n_epochs,optimizer,positive_labels,device="cpu"):

    for epoch in tqdm(range(n_epochs),total=n_epochs):
        #change the value to full if infrastructure supports full dataset
        for images,_ in tqdm(train_loader,total=len(train_loader)):
            image_org = images.to(device)
            image_aug = torch.flip(images,[1]).to(device)

            org_m = model(image_org)
            aug_m = model(image_aug)

            cl_loss = ConstructiveLearning(0.5,device).to(device)
            torch_loss = CLR_two_label(0.5,device,positive_labels).to(device)
            loss = torch_loss.forward(org_m)  # No augmented Image
            # loss = cl_loss.forward(org_m,aug_m)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'{loss}')

    torch.onnx.export(model,torch.randn(3,32,32).to(device), os.path.join(os.path.dirname(os.getcwd()),
                                                                             "models/model.onnx"))

if __name__ == "__main__":
    #testing
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data,test_data = load_data()
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
    positive_labels = torch.tensor([
        (0, 0), (0, 2), (0, 4),
        (1, 4), (1, 6), (1, 1),
        (2, 3),
        (3, 7),
        (4, 3),
        (7, 6),
        ])
    train_model(model,train_data,1,optimizer,positive_labels,device)


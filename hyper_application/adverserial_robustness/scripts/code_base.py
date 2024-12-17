import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torchattacks import FGSM, PGD

def load_model(model_path,pre_trained=False):
    """Return Model object if path is given else pretrained models is returned

    args: model path
    return model object
    """
    if not pre_trained:
        model = torch.load(model_path)
        model.eval()
        return model
    else:
        model = torch.hub.load("chenyaofo/pytorch-cifar-models",
                               "cifar10_resnet20",
                               pretrained=True)
        return model

def get_dataset(dataset_name, batch_size=64):
    """Load the specified dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if dataset_name == "MNIST":
        dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    elif dataset_name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
    else:
        st.error("Unsupported dataset.")
        return None

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def generate_adversarial_examples(model, dataloader, attack_type, epsilon):
    """Generate adversarial examples for the dataset."""
    attack = None
    if attack_type == "FGSM":
        attack = FGSM(model, eps=epsilon)
    elif attack_type == "PGD":
        attack = PGD(model, eps=epsilon)
    else:
        st.error("Unsupported attack type.")
        return None

    adv_examples = []
    adv_labels = []

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        adv_images = attack(images, labels)
        adv_examples.append(adv_images.cpu().detach())
        adv_labels.append(labels.cpu().detach())

    return torch.cat(adv_examples), torch.cat(adv_labels)

def visualize_examples(images, adv_images, labels):
    """Visualize original and adversarial examples."""
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    explanations = []

    for i in range(5):
        # Original image
        axes[0, i].imshow(np.transpose(images[i].numpy(), (1, 2, 0)))
        axes[0, i].set_title(f"Original: {labels[i].item()}")
        axes[0, i].axis("off")

        # Adversarial image
        axes[1, i].imshow(np.transpose(adv_images[i].numpy(), (1, 2, 0)))
        axes[1, i].set_title("Adversarial")
        axes[1, i].axis("off")

    st.pyplot(fig)

# Streamlit UI
def main():
    st.title("Adversarial Robustness - GenAI")

    with st.form("Input" , clear_on_submit=True):
        # Dataset Selection
        dataset_name = st.selectbox("Select Dataset", ["CIFAR10"])

        model_path = st.file_uploader("Upload your model (PyTorch .pth)", type=["pth"])


        batch_size = st.slider("Batch Size", min_value=1, max_value=128, value=64)

        attack_type = st.selectbox("Select Attack Type", ["FGSM", "PGD"])
        epsilon = st.slider("Epsilon (Attack Strength More the attack more vulnerable the model)", min_value=0.01, max_value=0.3, value=0.1)

        submit = st.form_submit_button()

    if submit:
        with st.spinner("Loading model..."):
            if model_path is not None:
                model = load_model(model_path, pre_trained=False)
                st.success("Model loaded successfully.")
            else:
                model = load_model(model_path, pre_trained=True)
                st.success("Pretrained Model has been loaded, since no model has been uploaded")


        with st.spinner("Loading dataset.."):
            dataloader = get_dataset(dataset_name, batch_size=batch_size)

        with st.spinner("Generating adverserial samples with the model. Please wait.."):
            if model and dataloader:
                adv_images, labels = generate_adversarial_examples(model, dataloader, attack_type, epsilon)
                st.success("Adversarial examples generated.")
                visualize_examples(next(iter(dataloader))[0], adv_images[:5], labels[:5])

            st.info(f"Adversarial image demonstrates how small perturbations (Changes to pixel values) "
                    f"can mislead the model, potentially causing a misclassification and expose sensitive information from the model")

    st.info(
        """
        ### Button Descriptions
        - **Attack Strength (Epsilon)**: Adjust the attack percentage. Higher values indicate stronger attacks.
        - **Batch Size** - Number of samples to be used per prediction instance.
        - **Attack Type**:
          - **FGSM**: Fast Gradient Sign Method (Default). Efficient and widely used attack.
          - **PGD**: Projected Gradient Descent. A more robust attack but doesn't cover all possibilities.
          
        ##### --- Note visualizations are limited to 5 --- 
        """
    )

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()

import streamlit as st
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from pre_processing import load_data
from model import SimpleCNN
from train import train_model

# Load trained model and dataset
@st.cache(allow_output_mutation=True)
def load_model_and_data():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("models/model.pth", map_location=device))  # Replace with your trained model path
    train_data, test_data = load_data()
    return model, train_data, test_data

# Visualize embeddings using t-SNE
def visualize_embeddings(model, data_loader, device="cpu"):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            outputs = model(images)
            embeddings.append(outputs.cpu().numpy())
            labels.append(targets.numpy())

    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)

    # Reduce dimensions to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    return reduced_embeddings, labels

# Streamlit UI
def streamlit_ui():
    st.title("SimpleCLR Model Visualization")
    st.sidebar.title("Navigation")
    options = ["Dataset Preview", "Embedding Visualization", "Training Metrics", "Original vs Augmented"]
    selection = st.sidebar.selectbox("Select a Visualization", options)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, train_data, test_data = load_model_and_data()

    if selection == "Dataset Preview":
        st.header("Dataset Preview")
        st.write("Explore the CIFAR-10 dataset with augmentations applied.")
        data_type = st.selectbox("Select Dataset", ["Train Data", "Test Data"])
        data_loader = train_data if data_type == "Train Data" else test_data

        # Show a batch of images
        images, labels = next(iter(data_loader))
        fig, axs = plt.subplots(4, 4, figsize=(10, 10))
        for i, ax in enumerate(axs.flatten()):
            ax.imshow(images[i].permute(1, 2, 0))
            ax.set_title(f"Label: {labels[i]}")
            ax.axis("off")
        st.pyplot(fig)

    elif selection == "Embedding Visualization":
        st.header("Embedding Visualization")
        data_type = st.selectbox("Select Dataset", ["Train Data", "Test Data"])
        data_loader = train_data if data_type == "Train Data" else test_data

        with st.spinner("Generating embeddings..."):
            reduced_embeddings, labels = visualize_embeddings(model, data_loader, device)

        # Plot embeddings
        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap="tab10", alpha=0.7)
        plt.colorbar(scatter, label="Class Labels")
        plt.title(f"{data_type} Embeddings (t-SNE Projection)")
        st.pyplot(plt)

    elif selection == "Training Metrics":
        st.header("Training Metrics Visualization")
        st.write("View training loss over epochs.")

        # Placeholder for loss curves
        # Replace with actual loss data from training
        epochs = np.arange(1, 11)
        clr_loss = np.random.rand(10)  # Replace with actual loss values

        plt.figure(figsize=(8, 4))
        plt.plot(epochs, clr_loss, label="CLR Loss", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        st.pyplot(plt)

    elif selection == "Original vs Augmented":
        st.header("Original vs Augmented Representations")
        st.write("Compare original and augmented image representations.")

        # Select a batch of images
        images, labels = next(iter(test_data))
        index = st.slider("Select Image Index", 0, len(images) - 1, 0)

        # Original and augmented images
        original_image = images[index].permute(1, 2, 0).numpy()
        augmented_image = torch.flip(images[index], [1]).permute(1, 2, 0).numpy()

        # Model embeddings
        with torch.no_grad():
            original_embedding = model(images[index].unsqueeze(0).to(device)).cpu().numpy()
            augmented_embedding = model(torch.flip(images[index].unsqueeze(0), [1]).to(device)).cpu().numpy()

        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption=f"Original Image (Label: {labels[index]})")
            st.write(f"Original Embedding: {original_embedding}")
        with col2:
            st.image(augmented_image, caption="Augmented Image (Flipped)")
            st.write(f"Augmented Embedding: {augmented_embedding}")

if __name__ == "__main__":
    streamlit_ui()

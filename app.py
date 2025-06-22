import streamlit as st
import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, 10)
        self.model = nn.Sequential(
            nn.Linear(100 + 10, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat((noise, self.label_emb(labels)), dim=1)
        img = self.model(x)
        return img.view(-1, 1, 28, 28)

@st.cache_resource
def load_model():
    try:
        model = Generator()
        model.load_state_dict(torch.load("generator.pth", map_location="cpu"))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

st.title("Handwritten Digit Generator")
digit = st.selectbox("Choose a digit (0–9)", list(range(10)))

model = load_model()
if model is None:
    st.stop()

if st.button("Generate"):
    try:
        z = torch.randn(5, 100)
        labels = torch.tensor([digit] * 5)
        with torch.no_grad():
            imgs = model(z, labels).detach().cpu()
        st.write("Generated Images:")
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                img = (imgs[i][0].numpy() + 1) / 2
                st.image(img, caption=f"Image {i+1}", use_column_width=True)
    except Exception as e:
        st.error(f"Error during generation: {e}")
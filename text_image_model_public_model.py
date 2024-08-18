import streamlit as st
import torch
from diffusers import DiffusionPipeline

# Streamlit setup for a simple user interface
st.title("Text-to-Image Generator")
st.write("Generate images from text prompts using Stable Diffusion.")

# Load the diffusion model
@st.cache_resource
def load_model():
    generator = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    generator.to("cuda" if torch.cuda.is_available() else "cpu")
    return generator

generator = load_model()

# Get user input
prompt = st.text_input("Enter a text prompt", "")

if st.button("Generate Image"):
    with st.spinner("Generating..."):
        # Generate the image
        image = generator(prompt).images[0]

        # Display the image
        st.image(image, caption=f"Generated Image for: {prompt}", use_column_width=True)

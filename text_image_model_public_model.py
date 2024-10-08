from diffusers import DiffusionPipeline
import streamlit as st
from diffusers import DiffusionPipeline

# Streamlit setup
st.title("Text-to-Image Generator")
st.write("Generate images from text prompts using Stable Diffusion.")

@st.cache_resource
def load_model():
    generator = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    generator.to("cuda" if torch.cuda.is_available() else "cpu")
    return generator

generator = load_model()

# Get user input
prompt = st.text_input("Enter a text prompt", "Show the car with accident")

if st.button("Generate Image"):
    with st.spinner("Generating..."):
        # Generate the image
        image = generator(prompt).images[0]

        # Display the image
        st.image(image, caption=f"Generated Image for: {prompt}", use_column_width=True)



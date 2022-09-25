import streamlit as st
import timm
import urllib
import torch

from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from typing import Dict

MODEL: str = "resnet18"

@st.cache
def load_model():
    model = timm.create_model(MODEL, pretrained=True)
    model.eval()

    # Download human-readable labels for ImageNet.
    # get the classnames
    url, filename = (
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
        "imagenet_classes.txt",
    )
    urllib.request.urlretrieve(url, filename)
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    return model, categories

@st.cache
def predict(inp_img: Image) -> Dict[str, float]:
    model, categories = load_model()

    config = resolve_data_config({}, model=MODEL)
    transform = create_transform(**config)

    img_tensor = transform(inp_img).unsqueeze(0)  # transform and add batch dimension

    # inference
    with torch.no_grad():
        out = model(img_tensor)
        probabilities = torch.nn.functional.softmax(out[0], dim=0)
        confidences = {categories[i]: float(probabilities[i]) for i in range(1000)}

    return confidences

def main():
    st.set_page_config(
        page_title="ResNet18 ImageNet Classifier",
        layout="centered",
        page_icon="🐍",
        initial_sidebar_state="expanded",
    )

    menu = ["Home", "About"]

    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.title("ResNet18 ImageNet Classifier")
        st.subheader("Upload an image to classify it with ResNet18")

        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "png", "jpeg"]
        )

        if st.button("Predict"):
            if uploaded_file is not None:
                image = Image.open(uploaded_file)

                st.image(image, caption="Uploaded Image", use_column_width=True)
                st.write("")

                try:
                    with st.spinner("Predicting..."):
                        predictions = predict(image)
                        # get key with highest value
                        prediction = max(predictions, key=predictions.get)
                        st.success(f"I think this is a {prediction}")
                except:
                    st.error("Something went wrong. Please try again.")
            else:
                st.warning("Please upload an image.")

    if choice == "About":
        st.title("About")

        # Model Card

        st.markdown(
            """
            ## Model Card for ResNet18

            ResNet model trained on imagenet-1k. It was introduced in the paper Deep Residual Learning for Image Recognition.

            ## Model Description

            ResNet introduced residual connections, they allow to train networks with an unseen number of layers (up to 1000). ResNet won the 2015 ILSVRC & COCO competition, one important milestone in deep computer vision.

            """
        )

if __name__ == "__main__":
    main()
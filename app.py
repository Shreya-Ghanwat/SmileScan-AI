

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2

# ✔ NEW IMPORTS FOR GRADCAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

st.set_page_config(page_title="SmileScan AI", layout="wide")

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
.stApp { background-color: #0d0d0d; color: white; }
.main-title { font-size: 52px; font-weight: 800; color: #ffffff; font-family: Georgia, serif; letter-spacing: 2px; margin-top: 20px; margin-bottom: 10px; text-align: center; }
.sub-title { font-size: 28px; color: #d9d9d9; margin-bottom: 35px; text-align: center; font-weight: 500; }
section[data-testid="stSidebar"] { background-color: #111111; }
.stRadio label, .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6 { color: white !important; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 100%; }
.stFileUploader { background-color: #1a1a1a; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# CLASS LABELS
# -----------------------------
oral_classes = [
    "Calculus", "Caries", "Hypodontia", "Mouth Ulcers", "Tooth Discoloration"
]

xray_classes = [
    "Cavity", "Impacted Tooth", "Abscess"
]

# -----------------------------
# DISEASE DETAILS
# -----------------------------
disease_info = {
    "Calculus": {
        "explanation": "Hardened plaque deposits on teeth that can lead to gum disease.",
        "remedy": "Brush twice daily, floss regularly, and rinse with mouthwash.",
        "treatment": "Professional dental cleaning and scaling."
    },
    "Caries": {
        "explanation": "Tooth decay caused by bacterial infection damaging the enamel.",
        "remedy": "Avoid sugary foods and maintain oral hygiene.",
        "treatment": "Dental filling, crown, or root canal depending on severity."
    },
    "Hypodontia": {
        "explanation": "A condition where one or more teeth are missing naturally.",
        "remedy": "Consult a dentist for long-term dental planning.",
        "treatment": "Braces, implants, bridges, or dentures."
    },
    "Mouth Ulcers": {
        "explanation": "Painful sores inside the mouth caused by stress, injury, or deficiency.",
        "remedy": "Avoid spicy foods and use saltwater rinse.",
        "treatment": "Topical gels, medicines, and vitamin supplements."
    },
    "Tooth Discoloration": {
        "explanation": "Change in tooth color due to stains, smoking, or enamel damage.",
        "remedy": "Avoid coffee, tobacco, and maintain brushing.",
        "treatment": "Teeth whitening, veneers, or polishing."
    },
    "Cavity": {
        "explanation": "A damaged area in the tooth caused by decay.",
        "remedy": "Avoid sweets and brush with fluoride toothpaste.",
        "treatment": "Dental filling or crown."
    },
    "Impacted Tooth": {
        "explanation": "A tooth trapped under the gum or bone, often wisdom teeth.",
        "remedy": "Avoid chewing on that side and use pain relief if needed.",
        "treatment": "Tooth extraction or surgery."
    },
    "Abscess": {
        "explanation": "A pocket of pus caused by bacterial infection near the tooth.",
        "remedy": "Rinse with warm salt water and avoid pressure on the area.",
        "treatment": "Drainage, root canal, or extraction."
    },
    "Normal": {
        "explanation": "No major visible disease detected.",
        "remedy": "Maintain regular brushing and flossing.",
        "treatment": "Routine dental checkups only."
    }
}

# -----------------------------
# IMAGE TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_xray_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(xray_classes))
    state_dict = torch.load("models/model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

@st.cache_resource
def load_oral_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(oral_classes))
    state_dict = torch.load("models/best_resnet50_dental_model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

xray_model = load_xray_model()
oral_model = load_oral_model()

# -----------------------------
# ✔ FIXED GRADCAM FUNCTION
# -----------------------------
def generate_gradcam(model, input_tensor, class_idx):
    target_layers = [model.layer4[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)

    targets = [ClassifierOutputTarget(class_idx)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    image = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())

    visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)

    return visualization

# -----------------------------
# UI LAYOUT
# -----------------------------
left_col, right_col = st.columns([1, 1])

with left_col:
    st.image("Assets/clinic.jpeg", width=500)

with right_col:
    st.markdown(
        """
        <div style="
            display: flex;
            flex-direction: column;
            justify-content: center;
            height: 60vh;
            text-align: center;
        ">
            <div class="main-title">SmileScan AI</div>
            <div class="sub-title">
                AI-powered dental disease prediction using X-rays and oral images.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )



diagnosis_type = st.radio("Choose Diagnosis Type", ["Dental X-Ray", "Oral Image"])
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "jfif"])

# -----------------------------
# PREDICTION
# -----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    uploaded_col, result_col = st.columns([1, 1])

    with uploaded_col:
        st.image(image, caption="Uploaded Image", width=300)

    input_tensor = transform(image).unsqueeze(0).to(device)

    if diagnosis_type == "Dental X-Ray":
        with torch.no_grad():
            output = xray_model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            if confidence.item() < 0.60:
                predicted_class = "Normal"
            else:
                predicted_class = xray_classes[predicted.item()]

        with uploaded_col:
            st.markdown("### Prediction Result")
            st.success(f"Prediction: {predicted_class}")

    else:
        with torch.no_grad():
            output = oral_model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(probabilities, 1)
            predicted_class = oral_classes[predicted.item()]

        # ✔ FIXED GRADCAM CALL
        gradcam_image = generate_gradcam(oral_model, input_tensor, predicted.item())

        with uploaded_col:
            st.markdown("### Prediction Result")
            st.success(f"Prediction: {predicted_class}")

        with result_col:
            st.image(gradcam_image, caption="Grad-CAM Visualization", width=350)

    info = disease_info[predicted_class]

    st.subheader("Disease Explanation")
    st.write(info["explanation"])

    st.subheader("Remedy Before Visiting Doctor")
    st.write(info["remedy"])

    st.subheader("Possible Treatment")
    st.write(info["treatment"])

    report = f"""
SmileScan AI Report

Predicted Disease: {predicted_class}
Explanation: {info['explanation']}
Remedy: {info['remedy']}
Treatment: {info['treatment']}
"""

    st.download_button(
        label="Download Report",
        data=report,
        file_name="dental_report.txt",
        mime="text/plain"
    )


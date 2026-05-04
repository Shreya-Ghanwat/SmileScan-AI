

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# GradCAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

st.set_page_config(page_title="SmileScan AI", layout="wide")

# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>
.stApp { background-color: #0d0d0d; color: white; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# CLASSES
# -----------------------------
oral_classes = ["Calculus", "Caries", "Hypodontia", "Mouth Ulcers", "Tooth Discoloration"]
xray_classes = ["Cavity", "Impacted Tooth", "Abscess"]

# -----------------------------
# DISEASE INFO (DETAILED)
# -----------------------------
disease_info = {
    "Calculus": {
        "explanation": "Hardened plaque deposits on teeth that can lead to gum disease.",
        "remedy": "Brush twice daily, floss regularly, and use antiseptic mouthwash.",
        "treatment": "Professional scaling and root planing is required to remove tartar. Severe cases need deep cleaning and follow-up care."
    },
    "Caries": {
        "explanation": "Tooth decay caused by bacteria damaging enamel.",
        "remedy": "Avoid sugary foods and maintain oral hygiene.",
        "treatment": "Fluoride treatment for early stage, fillings for moderate decay, and root canal with crown for severe damage."
    },
    "Hypodontia": {
        "explanation": "Missing teeth condition.",
        "remedy": "Consult dentist early.",
        "treatment": "Orthodontics, implants, bridges, or dentures depending on severity."
    },
    "Mouth Ulcers": {
        "explanation": "Painful sores inside mouth.",
        "remedy": "Avoid spicy foods and rinse with saltwater.",
        "treatment": "Topical gels, antiseptic rinses, and vitamin supplements."
    },
    "Tooth Discoloration": {
        "explanation": "Staining of teeth due to food, smoking, or aging.",
        "remedy": "Avoid tea, coffee, smoking.",
        "treatment": "Professional whitening, polishing, or veneers."
    },
    "Cavity": {
        "explanation": "Decay causing holes in teeth.",
        "remedy": "Brush properly and avoid sugar.",
        "treatment": "Fillings, crowns, or root canal depending on severity."
    },
    "Impacted Tooth": {
        "explanation": "Tooth stuck under gum.",
        "remedy": "Avoid pressure on that side.",
        "treatment": "Surgical extraction with proper post-care."
    },
    "Abscess": {
        "explanation": "Infection causing pus pocket.",
        "remedy": "Saltwater rinse.",
        "treatment": "Drainage, antibiotics, and root canal or extraction."
    },
    "Normal": {
        "explanation": "No disease detected.",
        "remedy": "Maintain hygiene.",
        "treatment": "Routine checkups."
    }
}

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_xray_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(xray_classes))
    model.load_state_dict(torch.load("models/model.pth", map_location=device))
    model.to(device).eval()
    return model

@st.cache_resource
def load_oral_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(oral_classes))
    model.load_state_dict(torch.load("models/best_resnet50_dental_model.pth", map_location=device))
    model.to(device).eval()
    return model

xray_model = load_xray_model()
oral_model = load_oral_model()

# -----------------------------
# GRADCAM
# -----------------------------
def generate_gradcam(model, input_tensor, class_idx):
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    targets = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    image = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())

    return show_cam_on_image(image, grayscale_cam, use_rgb=True)

# -----------------------------
# PDF
# -----------------------------
def create_pdf(predicted_class, confidence, info):
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("SmileScan AI Diagnosis Report", styles["Title"]))
    content.append(Spacer(1, 20))

    content.append(Paragraph(f"<b>Predicted Disease:</b> {predicted_class}", styles["Normal"]))
    content.append(Paragraph(f"<b>Confidence Score:</b> {confidence:.2f}%", styles["Normal"]))

    content.append(Spacer(1, 15))
    content.append(Paragraph("<b>About Disease:</b>", styles["Heading2"]))
    content.append(Paragraph(info["explanation"], styles["Normal"]))

    content.append(Spacer(1, 10))
    content.append(Paragraph("<b>Temporary Remedy:</b>", styles["Heading2"]))
    content.append(Paragraph(info["remedy"], styles["Normal"]))

    content.append(Spacer(1, 10))
    content.append(Paragraph("<b>Doctor Recommended Treatment:</b>", styles["Heading2"]))
    content.append(Paragraph(info["treatment"], styles["Normal"]))

    doc.build(content)

    with open("report.pdf", "rb") as f:
        return f.read()

# -----------------------------
# UI LAYOUT
# -----------------------------
left_col, right_col = st.columns([1, 1])

with left_col:
    st.image("Assets/clinic.jpeg", width=500)

with right_col:
    st.markdown("""
    <div style="text-align: center; margin-top: 40px;">
        <div style="font-size: 60px; font-weight: 800; color: white; margin-bottom: 25px;">
            SmileScan AI
        </div>
        <div style="font-size: 28px; color: white; margin-bottom: 40px;">
            AI-powered dental diagnosis
        </div>
    </div>
    """, unsafe_allow_html=True)

    diagnosis_type = st.radio(
        "Choose Diagnosis Type",
        ["Dental X-Ray", "Oral Image"],
        horizontal=True
    )

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "png", "jpeg", "jfif"]
    )

# -----------------------------
# PREDICTION
# -----------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=300)

    input_tensor = transform(image).unsqueeze(0).to(device)

    if diagnosis_type == "Dental X-Ray":
        model = xray_model
        classes = xray_classes
    else:
        model = oral_model
        classes = oral_classes

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    predicted_class = classes[pred.item()]
    confidence_score = confidence.item() * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence_score:.2f}%")

    if diagnosis_type == "Oral Image":
        cam_img = generate_gradcam(model, input_tensor, pred.item())
        st.image(cam_img, caption="Grad-CAM Visualization")

    info = disease_info[predicted_class]

    st.subheader("Explanation")
    st.write(info["explanation"])

    st.subheader("Remedy")
    st.write(info["remedy"])

    st.subheader("Treatment")
    st.write(info["treatment"])

    pdf_data = create_pdf(predicted_class, confidence_score, info)

    st.download_button(
        "Download PDF Report",
        pdf_data,
        file_name="SmileScan_Report.pdf",
        mime="application/pdf"
    )
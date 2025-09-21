<<<<<<< HEAD
import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title("Chest X-Ray Pneumonia Detection - Batch Upload & CSV Report")

model = models.efficientnet_v2_s(pretrained=False)
num_classes = 2
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load("ChestXRay(Pneumonia-Detection)Model_State_Dict.pth", map_location=device))
model = model.to(device)
model.eval()

idx_to_class = {0: "NORMAL", 1: "PNEUMONIA"}

predict_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def predict_image(model, image):
    input_tensor = predict_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][pred_class].item()
    return idx_to_class[pred_class], confidence

uploaded_files = st.file_uploader(
    "Upload Chest X-Ray images (multiple)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    results = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        label, confidence = predict_image(model, image)
        results.append({
            "image_name": uploaded_file.name,
            "prediction": label,
            "confidence": confidence
        })

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image, width=400)
        with col2:
            st.markdown(
                f"<h4>Image ID: {uploaded_file.name} <br><br>Prediction: {label} <br><br>Confidence: {confidence:.2f}</h2>",
                unsafe_allow_html=True
            )

    df_results = pd.DataFrame(results)
    st.write("### Prediction Report")
    st.dataframe(df_results)

    csv = df_results.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='pneumonia_predictions.csv',
        mime='text/csv'
    )
=======
import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title("Chest X-Ray Pneumonia Detection - Batch Upload & CSV Report")

model = models.efficientnet_v2_s(pretrained=False)
num_classes = 2
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load("ChestXRay(Pneumonia-Detection)Model_State_Dict.pth", map_location=device))
model = model.to(device)
model.eval()

idx_to_class = {0: "NORMAL", 1: "PNEUMONIA"}

predict_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def predict_image(model, image):
    input_tensor = predict_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][pred_class].item()
    return idx_to_class[pred_class], confidence

uploaded_files = st.file_uploader(
    "Upload Chest X-Ray images (multiple)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    results = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        label, confidence = predict_image(model, image)
        results.append({
            "image_name": uploaded_file.name,
            "prediction": label,
            "confidence": confidence
        })

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image, width=400)
        with col2:
            st.markdown(
                f"<h4>Image ID: {uploaded_file.name} <br><br>Prediction: {label} <br><br>Confidence: {confidence:.2f}</h2>",
                unsafe_allow_html=True
            )

    df_results = pd.DataFrame(results)
    st.write("### Prediction Report")
    st.dataframe(df_results)

    csv = df_results.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='pneumonia_predictions.csv',
        mime='text/csv'
    )
>>>>>>> b926fde9614b7669e8c97d3aad218c52d0059db2

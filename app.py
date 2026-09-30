import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageEnhance
import os
import pandas as pd
import altair as alt

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Brain Tumor Classifier PRO", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

@st.cache_resource
def load_resnet_model():
    model_path = 'brain_tumor_resnet.pth'
    if not os.path.exists(model_path):
        return None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Re-initialize the ResNet18 architecture
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        return "ERROR"

# --- SIDEBAR INTERFACE ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2854/2854061.png", width=100)
    st.title("Diagnostic AI v2")
    st.write("Powered by **ResNet-18 Transfer Learning** for high-precision medical image analysis.")
    
    st.markdown("### 📊 Supported Classes:")
    st.markdown("- 🟣 **Glioma**\n- 🔴 **Meningioma**\n- 🟡 **Pituitary**\n- 🟢 **No Tumor**")
    
    st.markdown("---")
    st.info("💡 **Tip:** Use the 'Image Inspector' tab to adjust the contrast and brightness of dark MRI scans.")

# --- MAIN INTERFACE ---
st.title("🧠 Advanced Brain Tumor MRI Analysis")
st.markdown("Upload a high-resolution MRI scan below. The AI will analyze the structural patterns and predict the tumor class while providing a full confidence distribution.")

model_data = load_resnet_model()

if model_data is None:
    st.error("⚠️ **ResNet Model weights not found!**")
    st.info("Please run `python train.py` in your terminal to train the new ResNet model. It will create `brain_tumor_resnet.pth`.")
elif model_data == "ERROR":
    st.error("⚠️ **Model architecture mismatch!**")
else:
    model, device = model_data
    
    # Patient Data Form
    with st.expander("📋 Enter Patient Details (Optional for Report)"):
        with st.form("patient_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                p_name = st.text_input("Patient Name / ID", "Anonymous_001")
            with c2:
                p_age = st.number_input("Age", min_value=1, max_value=120, value=45)
            with c3:
                p_date = st.date_input("Scan Date")
            st.form_submit_button("Save Patient Info")
    
    uploaded_file = st.file_uploader("📥 Drop your MRI scan here (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        
        # Load Image
        original_image = Image.open(uploaded_file).convert('RGB')
        
        # Create Interactive Tabs
        tab1, tab2, tab3 = st.tabs(["🩺 AI Diagnosis", "🎛️ Image Inspector", "🧠 About the Model"])
        
        with tab1:
            col_img, col_results = st.columns([1, 1.2], gap="large")
            
            with col_img:
                st.markdown("### 🖼️ Uploaded Scan")
                st.image(original_image, use_container_width=True, caption="Source: Uploaded File")
                
            # Preprocess Image for ResNet
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            
            input_tensor = transform(original_image).unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                confidence, predicted = torch.max(probabilities, 0)
                    
            predicted_class = classes[predicted.item()]
            conf_score = confidence.item() * 100
            
            # Results Column
            with col_results:
                st.markdown("### 🔬 AI Analysis Results")
                
                # Use metric for sleek UI
                if predicted_class == "No Tumor":
                    st.success(f"## ✅ {predicted_class}")
                else:
                    st.error(f"## ⚠️ Detected: {predicted_class} Tumor")
                    
                st.metric(label="AI Confidence Score", value=f"{conf_score:.2f}%")
                    
                st.markdown("---")
                st.markdown("#### 📊 Confidence Distribution")
                
                prob_df = pd.DataFrame({
                    'Tumor Type': classes,
                    'Probability (%)': (probabilities.cpu().numpy() * 100).round(2)
                })
                
                chart = alt.Chart(prob_df).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
                    x=alt.X('Tumor Type', sort='-y', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Probability (%)', scale=alt.Scale(domain=[0, 100])),
                    color=alt.condition(
                        alt.datum['Tumor Type'] == predicted_class,
                        alt.value('#ff4b4b'),     # Highlight
                        alt.value('#5c6c7b')      # Grey out
                    ),
                    tooltip=['Tumor Type', 'Probability (%)']
                ).properties(height=200)
                
                st.altair_chart(chart, use_container_width=True)
                
            # --- DOWNLOADABLE REPORT SECTION ---
            st.markdown("---")
            st.markdown("### 📄 Generate Medical Report")
            st.write("Generate a text-based summary report of this AI diagnosis for hospital records.")
            
            report_content = f"""=============================================
    BRAIN TUMOR MRI DIAGNOSTIC REPORT
=============================================
Patient ID / Name : {p_name}
Age               : {p_age}
Scan Date         : {p_date}
---------------------------------------------
AI PREDICTION     : {predicted_class.upper()}
CONFIDENCE SCORE  : {conf_score:.2f}%
---------------------------------------------
CLASS PROBABILITIES:
- Glioma      : {prob_df.loc[prob_df['Tumor Type'] == 'Glioma', 'Probability (%)'].values[0]:.2f}%
- Meningioma  : {prob_df.loc[prob_df['Tumor Type'] == 'Meningioma', 'Probability (%)'].values[0]:.2f}%
- No Tumor    : {prob_df.loc[prob_df['Tumor Type'] == 'No Tumor', 'Probability (%)'].values[0]:.2f}%
- Pituitary   : {prob_df.loc[prob_df['Tumor Type'] == 'Pituitary', 'Probability (%)'].values[0]:.2f}%
=============================================
DISCLAIMER: 
This report was generated automatically by a ResNet-18 
Artificial Intelligence prototype. It is for educational 
and preliminary screening purposes only. A licensed 
radiologist must verify all results.
============================================="""
            
            st.download_button(
                label="💾 Download Diagnostic Report (.txt)",
                data=report_content,
                file_name=f"Diagnosis_Report_{p_name.replace(' ', '_')}.txt",
                mime="text/plain",
                use_container_width=True
            )
                
        with tab2:
            st.markdown("### 🎛️ Adjust Image Visibility")
            st.write("Use the sliders to inspect dark or low-contrast regions of the MRI.")
            
            col_sliders, col_preview = st.columns([1, 1])
            
            with col_sliders:
                contrast_factor = st.slider("Contrast", 0.5, 3.0, 1.0, step=0.1)
                brightness_factor = st.slider("Brightness", 0.5, 3.0, 1.0, step=0.1)
                
            with col_preview:
                enhancer_contrast = ImageEnhance.Contrast(original_image)
                img_c = enhancer_contrast.enhance(contrast_factor)
                
                enhancer_brightness = ImageEnhance.Brightness(img_c)
                final_img = enhancer_brightness.enhance(brightness_factor)
                
                st.image(final_img, use_container_width=True, caption="Enhanced MRI")
                
        with tab3:
            st.markdown("### 🧠 Transfer Learning with ResNet-18")
            st.write(
                "This application uses **Transfer Learning**, a technique where an AI model "
                "that was originally trained on millions of images (ImageNet) is re-trained "
                "on a specific medical dataset."
            )
            st.write(
                "**Why ResNet-18?** \n"
                "- It contains 18 deep layers, allowing it to see microscopic textures.\n"
                "- It uses 'Residual Connections', solving the problem of deep networks forgetting features.\n"
                "- It massively outperforms custom-built CNNs in medical imaging tasks, eliminating the confusion between Pituitary and Meningioma tumors."
            )

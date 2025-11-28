import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="Chili Doctor AI",
    page_icon="🌶️",
    layout="centered"
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600&display=swap" rel="stylesheet">

<style>

    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Prompt', sans-serif !important;
    }

    /* --- Background ไล่สีแบบ HTML --- */
    .stApp {
        background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%) !important;
        background-attachment: fixed !important;
        min-height: 100vh !important;
        padding: 20px !important;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    /* --- Glass Card เหมือน HTML (95% similar) --- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border-radius: 24px !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2) !important;
        max-width: 480px !important;
        margin: auto !important;
        padding: 30px 25px !important;
        animation: fadeUp 0.8s ease-out;
    }

    /* Pulse Icon (เหมือน HTML) */
    .app-icon {
        width: 100px;
        height: 100px;
        background: linear-gradient(45deg, #ff9a9e, #fad0c4);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 50px;
        margin: 0 auto 15px;
        box-shadow: 0 4px 15px rgba(255, 75, 43, 0.3);
        animation: pulse 2s infinite;
    }

    /* Header Typography */
    h1 {
        font-weight: 600 !important;
        color: #333 !important;
        font-size: 1.9rem !important;
        margin-bottom: 0 !important;
        text-align: center;
    }

    .subtitle {
        color: #d32f2f !important;
        font-weight: 500;
        font-size: 0.9rem !important;
        letter-spacing: 1px;
        text-transform: uppercase;
        text-align: center;
        margin-top: -5px;
    }

    /* Button เหมือน HTML */
    div.stButton > button {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%) !important;
        border: none !important;
        color: white !important;
        padding: 15px 40px !important;
        border-radius: 50px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(255, 65, 108, 0.4) !important;
        transition: all 0.3s ease;
        width: 100%;
    }

    div.stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(255, 65, 108, 0.6) !important;
    }

    /* File uploader – minimal white glass */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #ffffff !important;
        border: 2px dashed #FF8A80 !important;
        border-radius: 16px !important;
        padding: 25px !important;
    }

    /* Keyframe for fadeUp */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(40px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Pulse animation */
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255,75,43,0.4); }
        70% { transform: scale(1.05); box-shadow: 0 0 0 10px rgba(255,75,43,0); }
        100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255,75,43,0); }
    }

    #MainMenu, header, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 3. โหลดโมเดล ---
@st.cache_resource
def load_model():
    filename = 'efficientnetb4_model.h5'
    if not os.path.exists(filename):
        pass # Handle download here if needed
    try:
        return tf.keras.models.load_model(filename)
    except:
        return None

def import_and_predict(image_data, model):
    size = (300, 300)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype(np.float32)
    data = np.ndarray(shape=(1, 300, 300, 3), dtype=np.float32)
    data[0] = img_array
    return model.predict(data)

# --- 4. ส่วนแสดงผล UI ---

model = load_model()

# เริ่มสร้าง Card Container
with st.container(border=True):
    
    # Header Section
    st.markdown("""
        <div style="text-align: center; padding-top: 10px;">
            <div class="floating-icon">🌶️</div>
            <h1>Chili Doctor AI</h1>
            <div class="subtitle">ระบบผู้เชี่ยวชาญตรวจวินิจฉัยโรคพริกอัจฉริยะ</div>
            <span class="badge">EfficiencyNetB4 Model</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Upload Section
    file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    if file is not None:
        image = Image.open(file)
        
        # Display Image (Centered & Styled)
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([0.5, 5, 0.5])
        with col2:
            st.image(image, use_container_width=True)
        
        # File Info
        size_kb = file.size / 1024
        st.markdown(f"""
            <div style="text-align: center; margin-top: 15px; font-size: 0.85rem; color: #888;">
                <span style="background: #f5f5f5; padding: 4px 10px; border-radius: 10px;">
                    📎 {file.name} • {size_kb:.1f} KB
                </span>
            </div>
        """, unsafe_allow_html=True)
            
        # Button
        if st.button("🔍 START DIAGNOSIS"):
            if model is None:
                st.error("⚠️ Model file not found.")
            else:
                with st.spinner('AI is analyzing...'):
                    predictions = import_and_predict(image, model)
                    class_names = ['Healthy', 'Leaf Curl', 'Leaf Spot', 'Whitefly', 'Yellow']
                    class_index = np.argmax(predictions)
                    result_class = class_names[class_index]
                    confidence = np.max(predictions) * 100

                st.markdown("<hr style='margin: 25px 0; border: 0; border-top: 1px dashed #ddd;'>", unsafe_allow_html=True)
                
                # Result Display (สวยงามขึ้น)
                st.markdown(f"""
                    <div style="text-align: center;">
                        <div style="color: #999; font-size: 0.85rem; letter-spacing: 1px; text-transform: uppercase;">DIAGNOSIS RESULT</div>
                        <h2 style="color: #d32f2f; margin: 10px 0; font-size: 2rem; font-weight: 700;">{result_class.upper()}</h2>
                        <div style="margin-top: 5px;">
                            <span style="background: #e8f5e9; color: #2e7d32; padding: 5px 15px; border-radius: 20px; font-weight: 600; font-size: 0.9rem;">
                                Confidence: {confidence:.2f}%
                            </span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # Treatment Logic
                treatment_title = "Recommendation"
                treatment_text = ""
                bg_color = "#fff8e1" # สีเหลืองอ่อนพาสเทล
                icon = "💡"
                
                if result_class == 'Healthy':
                    treatment_text = "ต้นพริกแข็งแรงดีมาก! แนะนำให้ดูแลรดน้ำและใส่ปุ๋ยตามปกติต่อไป"
                    bg_color = "#e8f5e9" # เขียวพาสเทล
                    icon = "🌿"
                elif result_class == 'Leaf Curl':
                    treatment_text = "พบโรคใบหงิก แนะนำให้กำจัดวัชพืชรอบแปลง และใช้สารสกัดสะเดาฉีดพ่นเพื่อไล่แมลงพาหะ"
                    icon = "🍂"
                elif result_class == 'Leaf Spot':
                    treatment_text = "พบโรคใบจุด ตัดแต่งใบที่เป็นโรคไปเผาทำลายทันที และฉีดพ่นสารป้องกันเชื้อรา"
                    icon = "🌑"
                elif result_class == 'Whitefly':
                    treatment_text = "พบแมลงหวี่ขาว ใช้กับดักกาวเหนียวสีเหลือง หรือฉีดพ่นน้ำหมักสมุนไพร"
                    icon = "🪰"
                elif result_class == 'Yellow':
                    treatment_text = "พบอาการใบเหลือง อาจขาดธาตุอาหาร ตรวจสอบสภาพดินและเติมปุ๋ยบำรุง"
                    icon = "🟡"
                
                # Treatment Box (Modern Style)
                st.markdown(f"""
                    <div style="background-color: {bg_color}; padding: 25px; border-radius: 20px; margin-top: 25px; text-align: left; border-left: 5px solid rgba(0,0,0,0.1);">
                        <div style="font-weight: 600; color: #333; margin-bottom: 8px; font-size: 1.1rem;">
                            {icon} {treatment_title}
                        </div>
                        <div style="color: #444; font-size: 0.95rem; line-height: 1.6;">
                            {treatment_text}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        Computer Research Project • UBRU<br>
        <span style="opacity: 0.7; font-size: 0.7rem;">Designed by WhiteCat Team</span>
    </div>
""", unsafe_allow_html=True)
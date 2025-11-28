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

# --- 2. 🎨 CSS ตกแต่ง (Clean & Minimal Style) ---
st.markdown("""
<style>
    /* นำเข้าฟอนต์ Prompt */
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
    
    /* บังคับฟอนต์ทั้งหน้า */
    html, body, [class*="css"] {
        font-family: 'Prompt', sans-serif;
    }
    
    /* 1. พื้นหลังหลัก: สีส้มแดง Gradient สวยๆ */
    .stApp, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%) !important;
    }

    /* 2. ปรับแต่ง "การ์ดสีขาว" (Card) */
    /* เป้าหมายคือ st.container(border=True) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border-radius: 24px !important;
        padding: 40px !important;
        box-shadow: 0 20px 60px rgba(0,0,0,0.2) !important;
        border: none !important;
        margin-bottom: 20px;
    }
    
    /* 3. จัดการ Typography (ตัวหนังสือ) ให้อ่านง่าย */
    .app-subtitle {
        color: #FF4B2B;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        text-align: center;
        margin-bottom: 5px;
    }
    
    .app-title {
        color: #333333;
        font-weight: 800;
        font-size: 2.5rem;
        margin: 0;
        padding: 0;
        text-align: center;
        letter-spacing: -1px;
        line-height: 1.2;
    }
    
    .app-desc {
        color: #555555;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 400;
        margin-top: 15px;
        line-height: 1.6;
    }
    
    .app-note {
        color: #FF6B6B;
        font-size: 0.95rem;
        text-align: center;
        font-weight: 500;
        margin-bottom: 30px;
    }
    
    /* 4. ปรับแต่งช่องอัปโหลดไฟล์ (File Uploader) */
    /* พื้นที่ Dropzone */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #F9FAFB !important;
        border: 2px dashed #E5E7EB !important;
        border-radius: 16px !important;
        padding: 30px !important;
        transition: all 0.2s ease-in-out;
    }
    
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #FF4B2B !important;
        background-color: #FFF5F5 !important;
    }
    
    /* ซ่อนไอคอนเล็กๆ เดิมของ Streamlit */
    [data-testid="stFileUploaderDropzone"] div svg {
        display: none;
    }
    
    /* ข้อความใน Dropzone */
    [data-testid="stFileUploaderDropzone"] div div::before {
        content: "📂 คลิก หรือ ลากไฟล์รูปภาพมาวางที่นี่";
        font-size: 1.1rem;
        color: #6B7280;
        font-weight: 500;
        display: block;
        margin-bottom: 5px;
        text-align: center;
    }
    
    /* 5. ปุ่มกด (Button) */
    div.stButton > button {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 24px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        box-shadow: 0 4px 6px -1px rgba(255, 75, 43, 0.3) !important;
        margin-top: 15px !important;
        transition: all 0.2s !important;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(255, 75, 43, 0.4) !important;
    }
    
    /* ไอคอนพริก */
    .icon-container {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .main-icon {
        font-size: 60px;
        background: #FFF5F5;
        border-radius: 50%;
        width: 100px;
        height: 100px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 10px 30px rgba(255, 75, 43, 0.15);
    }
    
    /* ซ่อน Header/Footer */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* จัดการการ์ดผลลัพธ์ */
    .result-card {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. โหลดโมเดล (ฟังก์ชันเดิม) ---
@st.cache_resource
def load_model():
    filename = 'efficientnetb4_model.h5'
    if not os.path.exists(filename):
        file_id = '1tURhAR8mXLAgnuU3EULswpcFGxnalWAV'
        url = f'https://drive.google.com/uc?id={file_id}'
        with st.status("⏳ กำลังดาวน์โหลดโมเดล...", expanded=True) as status:
            try:
                import gdown
                gdown.download(url, filename, quiet=False)
                if os.path.exists(filename):
                    status.update(label="✅ เสร็จสิ้น!", state="complete", expanded=False)
                else:
                    return None
            except:
                return None
    try:
        return tf.keras.models.load_model(filename)
    except:
        return None

# ฟังก์ชันทำนาย
def import_and_predict(image_data, model):
    size = (300, 300)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype(np.float32)
    data = np.ndarray(shape=(1, 300, 300, 3), dtype=np.float32)
    data[0] = img_array
    return model.predict(data)

# --- 4. ส่วนแสดงผล (UI) ---

model = load_model()

# ==========================================
# ⬜ ส่วน Input (การ์ดใบที่ 1)
# ==========================================
with st.container(border=True): # สร้างกรอบขาว
    
    # ไอคอน + หัวข้อ
    st.markdown("""
        <div class="icon-container">
            <div class="main-icon">🌶️</div>
        </div>
        <div class="app-subtitle">AI Expert System</div>
        <div class="app-title">Chili Doctor AI</div>
        
        <div class="app-desc">
            ระบบผู้เชี่ยวชาญปัญญาประดิษฐ์เพื่อวินิจฉัยโรคของพริกจากใบ
        </div>
        <div class="app-note">
            (กรุณาอัปโหลดรูปภาพที่เห็นใบพริกชัดเจน)
        </div>
    """, unsafe_allow_html=True)

    # ช่องอัปโหลดไฟล์ (อยู่ในกรอบขาวเดียวกัน)
    file = st.file_uploader("", type=["jpg", "png", "jpeg"])

# ==========================================
# ⬜ ส่วน Result (การ์ดใบที่ 2 - แสดงเมื่อมีไฟล์)
# ==========================================
if file is not None:
    # สร้างกรอบขาวอีกอันแยกออกมา
    with st.container(border=True):
        image = Image.open(file)
        
        # จัดรูปให้อยู่ตรงกลาง
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, use_container_width=True)
        
        # ปุ่มกดวิเคราะห์
        if st.button("🔍 วิเคราะห์โรคเดี๋ยวนี้"):
            if model is None:
                st.error("❌ ไม่สามารถโหลดโมเดลได้")
            else:
                with st.spinner('🤖 AI กำลังประมวลผล...'):
                    predictions = import_and_predict(image, model)
                    class_names = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellow']
                    class_index = np.argmax(predictions)
                    result_class = class_names[class_index]
                    confidence = np.max(predictions) * 100

                # แสดงเส้นคั่น
                st.markdown("<hr style='margin: 30px 0; border-top: 1px solid #eee;'>", unsafe_allow_html=True)
                
                # แสดงผลลัพธ์
                st.markdown(f"""
                    <div style="text-align: center; margin-bottom: 25px;">
                        <div style="color: #6B7280; font-size: 1rem; font-weight: 500; margin-bottom: 5px;">ผลการวิเคราะห์</div>
                        <h1 style="color: #10B981; font-size: 2.8rem; margin: 0; font-weight: 800;">{result_class}</h1>
                        <div style="color: #9CA3AF; font-size: 1rem; margin-top: 5px;">ความมั่นใจ: <b>{confidence:.2f}%</b></div>
                    </div>
                """, unsafe_allow_html=True)

                # คำแนะนำ (Card ย่อย)
                treatment_text = ""
                bg_color = "#FEF3C7"
                text_color = "#92400E"
                border_color = "#FDE68A"
                icon = "⚠️"

                if result_class == 'healthy':
                    treatment_text = "ต้นพริกแข็งแรงดี! ไม่พบร่องรอยโรค หมั่นดูแลรดน้ำและใส่ปุ๋ยตามปกติ"
                    bg_color = "#D1FAE5"
                    text_color = "#065F46"
                    border_color = "#A7F3D0"
                    icon = "🌿"
                elif result_class == 'leaf curl':
                    treatment_text = "โรคใบหงิกมักเกิดจากแมลงหวี่ขาว ให้กำจัดวัชพืชและใช้สารสกัดสะเดา หรือเชื้อราเมตาไรเซียมฉีดพ่น"
                elif result_class == 'leaf spot':
                    treatment_text = "โรคใบจุดตากบ เกิดจากเชื้อรา ให้ตัดแต่งใบที่เป็นโรคเผาทำลาย และฉีดพ่นสารป้องกันเชื้อรา"
                elif result_class == 'whitefly':
                     treatment_text = "พบแมลงหวี่ขาว ให้ใช้กับดักกาวเหนียวสีเหลือง หรือฉีดพ่นน้ำหมักสมุนไพร"
                elif result_class == 'yellow':
                     treatment_text = "อาการใบเหลือง อาจเกิดจากการขาดสารอาหาร หรือไวรัส ควรตรวจสอบดินและใส่ปุ๋ยบำรุง"
                
                st.markdown(f"""
                    <div style="background-color: {bg_color}; color: {text_color}; border: 1px solid {border_color}; padding: 25px; border-radius: 16px; text-align: left; font-size: 1.1rem; line-height: 1.6;">
                        <strong style="display:block; margin-bottom:10px; font-size:1.2rem;">{icon} คำแนะนำ:</strong>
                        {treatment_text}
                    </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 40px; color: rgba(255,255,255,0.9); font-size: 0.9rem; font-weight: 300;">
        โครงงานวิจัยทางคอมพิวเตอร์ • มหาวิทยาลัยราชภัฏอุบลราชธานี<br>
        พัฒนาโดย: แมวสีขาวเทา และผองเพื่อน
    </div>
""", unsafe_allow_html=True)
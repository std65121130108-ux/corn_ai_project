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

# --- 2. 🎨 CSS ตกแต่ง (White Card Edition) ---
st.markdown("""
<style>
    /* นำเข้าฟอนต์ Prompt */
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600;700&display=swap');
    
    /* บังคับฟอนต์ทั้งหน้า */
    html, body, [class*="css"] {
        font-family: 'Prompt', sans-serif;
    }
    
    /* 1. พื้นหลังหลัก (Background): สีส้มแดงไล่เฉด */
    .stApp, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%) !important;
    }

    /* 2. ปรับแต่ง "กรอบสีขาว" (Card) ที่สร้างจาก st.container(border=True) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important; /* สีขาวทึบ 100% */
        border-radius: 20px !important;
        padding: 30px !important;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3) !important; /* เงาเข้มขึ้นเพื่อให้ลอยเด่น */
        border: none !important;
        margin-bottom: 25px;
    }
    
    /* แก้ไขสีพื้นหลังของกล่องภายใน (กัน Streamlit ใส่สีเทาทับ) */
    [data-testid="stVerticalBlock"] {
        background-color: transparent !important;
    }

    /* 3. จัดการข้อความ (Typography) */
    .title-text {
        color: #111111 !important; /* สีดำเกือบสนิท */
        font-weight: 800 !important;
        font-size: 2.5rem;
        margin: 0;
        padding: 0;
        text-align: center;
        letter-spacing: -1px;
    }
    .subtitle-text {
        color: #FF4B2B !important;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        text-align: center;
        margin-bottom: 5px;
    }
    .desc-text {
        color: #555555 !important;
        text-align: center;
        font-size: 1.1rem;
        margin-top: 15px;
        margin-bottom: 25px;
        line-height: 1.6;
    }
    
    /* 4. ปรับช่องอัปโหลดไฟล์ (File Uploader) */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #F8F9FA !important; /* สีเทาอ่อนมากๆ ตัดกับพื้นขาว */
        border: 2px dashed #FF6B6B !important; /* เส้นประสีแดงอ่อน */
        border-radius: 12px !important;
        padding: 40px 20px !important;
    }
    /* ข้อความใน Dropzone */
    [data-testid="stFileUploaderDropzone"] div div::before {
        content: "📂 ลากไฟล์รูปภาพวางที่นี่";
        font-size: 1.2rem;
        color: #666;
        font-weight: 600;
        display: block;
        margin-bottom: 10px;
    }
    [data-testid="stFileUploaderDropzone"] div div {
        color: #888 !important;
    }
    
    /* 5. ปุ่มกด (Button) */
    div.stButton > button {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 12px 35px !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        box-shadow: 0 10px 20px rgba(255, 75, 43, 0.4) !important;
        margin-top: 10px;
    }
    div.stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(255, 75, 43, 0.6) !important;
    }
    
    /* ไอคอนพริก */
    .icon-container {
        text-align: center;
        margin-bottom: 15px;
    }
    .main-icon {
        font-size: 70px;
        background: #fff;
        border-radius: 50%;
        padding: 10px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    
    /* ซ่อน Header/Footer */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
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
# 🟩 การ์ดรับข้อมูล (Input Card) - พื้นหลังขาวทึบ
# ==========================================
with st.container(border=True):
    # ไอคอน
    st.markdown('<div class="icon-container"><span class="main-icon">🌶️</span></div>', unsafe_allow_html=True)
    
    # หัวข้อ
    st.markdown('<div class="subtitle-text">AI Expert System</div>', unsafe_allow_html=True)
    st.markdown('<div class="title-text">Chili Doctor AI</div>', unsafe_allow_html=True)
    
    # คำอธิบาย
    st.markdown("""
        <div class="desc-text">
            ระบบผู้เชี่ยวชาญปัญญาประดิษฐ์เพื่อวินิจฉัยโรคของพริกจากใบ<br>
            <span style="color: #FF4B2B; font-weight: 500;">(กรุณาอัปโหลดรูปภาพที่เห็นใบพริกชัดเจน)</span>
        </div>
    """, unsafe_allow_html=True)

    # ช่องอัปโหลดไฟล์
    file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    # ข้อความแนะนำด้านล่าง (จะแสดงเมื่อยังไม่เลือกไฟล์)
    if file is None:
        st.markdown("""
            <div style="text-align: center; color: #999; margin-top: 15px; font-size: 0.9rem;">
                👆 เลือกไฟล์รูปภาพ (.jpg, .png) จากเครื่องของคุณ
            </div>
        """, unsafe_allow_html=True)

# ==========================================
# 🟩 การ์ดแสดงผล (Result Card)
# ==========================================
if file is not None:
    # สร้างกรอบขาวอีกอันสำหรับผลลัพธ์
    with st.container(border=True):
        image = Image.open(file)
        
        # จัดกึ่งกลางรูปภาพ
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, use_container_width=True, caption="รูปภาพที่เลือก")
        
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

                # แสดงผลแบบสวยงาม
                st.markdown("<hr style='margin: 20px 0; border-top: 1px solid #eee;'>", unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div style="text-align: center;">
                        <h3 style="color: #333; margin-bottom: 5px;">ผลการวิเคราะห์</h3>
                        <h1 style="color: #28a745; font-size: 2.5rem; margin: 0; font-weight: 800;">{result_class}</h1>
                        <p style="color: #777;">ความมั่นใจ: <b>{confidence:.2f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)

                if result_class == 'healthy':
                    st.success("✅ **คำแนะนำ:** ต้นพริกแข็งแรงดี! ไม่พบร่องรอยโรค หมั่นดูแลรดน้ำและใส่ปุ๋ยตามปกติ")
                elif result_class == 'leaf curl':
                    st.warning("⚠️ **คำแนะนำ:** โรคใบหงิกมักเกิดจากแมลงหวี่ขาว ให้กำจัดวัชพืชและใช้สารสกัดสะเดา")
                elif result_class == 'leaf spot':
                    st.warning("⚠️ **คำแนะนำ:** โรคใบจุดตากบ เกิดจากเชื้อรา ให้ตัดแต่งใบที่เป็นโรคเผาทำลาย และฉีดพ่นสารป้องกันเชื้อรา")
                elif result_class == 'whitefly':
                     st.warning("⚠️ **คำแนะนำ:** พบแมลงหวี่ขาว ให้ใช้กับดักกาวเหนียวสีเหลือง หรือฉีดพ่นน้ำหมักสมุนไพร")
                elif result_class == 'yellow':
                     st.warning("⚠️ **คำแนะนำ:** อาการใบเหลือง อาจเกิดจากการขาดสารอาหาร หรือไวรัส ควรตรวจสอบดินและใส่ปุ๋ยบำรุง")

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 40px; color: rgba(255,255,255,0.8); font-size: 0.8rem;">
        โครงงานวิจัยทางคอมพิวเตอร์ • มหาวิทยาลัยราชภัฏอุบลราชธานี<br>
        พัฒนาโดย: แมวสีขาวเทา และผองเพื่อน
    </div>
""", unsafe_allow_html=True)
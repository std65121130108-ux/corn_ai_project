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

# --- 2. 🎨 CSS ตกแต่ง (เน้นการสร้าง Card สีขาว) ---
st.markdown("""
<style>
    /* นำเข้าฟอนต์ Prompt */
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600;700&display=swap');
    
    /* บังคับฟอนต์ทั้งหน้า */
    html, body, [class*="css"] {
        font-family: 'Prompt', sans-serif;
    }
    
    /* 1. พื้นหลังหลัก (Background): สีส้มแดงไล่เฉด */
    .stApp {
        background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%) !important;
    }

    /* 2. ซ่อน Header/Footer เดิมของ Streamlit */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* 3. ปรับแต่ง "กรอบสีขาว" (Card) ที่เราจะใช้ st.container(border=True) สร้าง */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important; /* พื้นหลังสีขาวทึบ */
        border-radius: 20px !important;       /* มุมโค้ง */
        padding: 40px !important;             /* เว้นขอบด้านใน */
        box-shadow: 0 10px 30px rgba(0,0,0,0.2) !important; /* เงาลอยมีมิติ */
        border: none !important;              /* ลบเส้นขอบสีเทาเดิมออก */
        margin-bottom: 20px;
    }

    /* 4. จัดการข้อความ (Typography) ในการ์ด */
    .title-text {
        color: #333333;
        font-weight: 700;
        font-size: 2.2rem;
        margin: 0;
        padding: 0;
        text-align: center;
    }
    .subtitle-text {
        color: #FF4B2B;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        text-align: center;
        margin-bottom: 5px;
    }
    .desc-text {
        color: #666666;
        text-align: center;
        font-size: 1rem;
        margin-top: 10px;
        margin-bottom: 30px;
        line-height: 1.6;
    }
    
    /* 5. ปรับช่องอัปโหลดไฟล์ (File Uploader) */
    [data-testid="stFileUploader"] {
        width: 100%;
        padding-top: 10px;
    }
    /* พื้นที่ Dropzone */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #f8f9fa !important;
        border: 2px dashed #FF4B2B !important;
        border-radius: 15px !important;
        padding: 30px 10px !important; /* เพิ่มความสูงให้ดูง่าย */
        min-height: 150px;
        align-items: center;
        justify-content: center;
        display: flex;
        flex-direction: column;
    }
    /* ข้อความใน Dropzone */
    [data-testid="stFileUploaderDropzone"] div div::before {
        content: "📂 ลากไฟล์มาวาง หรือคลิกเพื่อเลือกรูปภาพ";
        font-size: 1rem;
        color: #555;
        font-weight: 500;
    }
    
    /* 6. ปุ่มกด (Button) */
    div.stButton > button {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 12px 30px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        box-shadow: 0 5px 15px rgba(255, 75, 43, 0.4) !important;
        margin-top: 15px;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(255, 75, 43, 0.6) !important;
    }
    
    /* ไอคอนพริก */
    .icon-container {
        text-align: center;
        margin-bottom: 20px;
    }
    .main-icon {
        font-size: 60px;
        background: #fff0f0;
        border-radius: 50%;
        padding: 20px;
        border: 4px solid #fff;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
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

# โหลดโมเดลรอไว้
model = load_model()

# ==========================================
# 🟩 ส่วนที่ 1: การ์ดรับข้อมูล (Input Card)
# ==========================================
# ใช้ container(border=True) เพื่อสร้างกรอบ และ CSS ด้านบนจะเปลี่ยนมันเป็นสีขาวให้เอง
with st.container(border=True):
    # ไอคอน
    st.markdown('<div class="icon-container"><span class="main-icon">🌶️</span></div>', unsafe_allow_html=True)
    
    # หัวข้อ (เรียงตามที่คุณสั่ง)
    st.markdown('<div class="subtitle-text">AI Expert System</div>', unsafe_allow_html=True)
    st.markdown('<div class="title-text">Chili Doctor AI</div>', unsafe_allow_html=True)
    
    # คำอธิบาย
    st.markdown("""
        <div class="desc-text">
            ระบบผู้เชี่ยวชาญปัญญาประดิษฐ์เพื่อวินิจฉัยโรคของพริกจากใบ<br>
            <span style="font-size: 0.9rem; color: #FF4B2B;">(กรุณาอัปโหลดรูปภาพที่เห็นใบพริกชัดเจน)</span>
        </div>
    """, unsafe_allow_html=True)

    # ช่องอัปโหลดไฟล์ (อยู่ในกรอบขาว)
    file = st.file_uploader("", type=["jpg", "png", "jpeg"])

# ==========================================
# 🟩 ส่วนที่ 2: การ์ดแสดงผล (Result Card)
# ==========================================
# จะแสดงก็ต่อเมื่อมีไฟล์ถูกเลือกแล้วเท่านั้น
if file is not None:
    # สร้างกรอบขาวอีกอันสำหรับผลลัพธ์
    with st.container(border=True):
        image = Image.open(file)
        
        # แสดงรูปภาพ
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, use_container_width=True, caption="รูปภาพของคุณ")
        
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
                
                # ชื่อโรค
                st.markdown(f"""
                    <div style="text-align: center;">
                        <h3 style="color: #333; margin-bottom: 5px;">ผลการวิเคราะห์</h3>
                        <h1 style="color: #28a745; font-size: 2.5rem; margin: 0;">{result_class}</h1>
                        <p style="color: #777;">ความมั่นใจ: <b>{confidence:.2f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)

                # คำแนะนำ (ใช้ st.info, st.warning, st.success ตกแต่ง)
                if result_class == 'healthy':
                    st.success("✅ **คำแนะนำ:** ต้นพริกแข็งแรงดี! ไม่พบร่องรอยโรค หมั่นดูแลรดน้ำและใส่ปุ๋ยตามปกติ")
                elif result_class == 'leaf curl':
                    st.warning("⚠️ **คำแนะนำ:** โรคใบหงิกมักเกิดจากแมลงหวี่ขาว ให้กำจัดวัชพืชและใช้สารสกัดสะเดา หรือเชื้อราเมตาไรเซียมฉีดพ่น")
                elif result_class == 'leaf spot':
                    st.warning("⚠️ **คำแนะนำ:** โรคใบจุดตากบ เกิดจากเชื้อรา ให้ตัดแต่งใบที่เป็นโรคเผาทำลาย และฉีดพ่นสารป้องกันเชื้อรา")
                elif result_class == 'whitefly':
                     st.warning("⚠️ **คำแนะนำ:** พบแมลงหวี่ขาว ให้ใช้กับดักกาวเหนียวสีเหลือง หรือฉีดพ่นน้ำหมักสมุนไพร")
                elif result_class == 'yellow':
                     st.warning("⚠️ **คำแนะนำ:** อาการใบเหลือง อาจเกิดจากการขาดสารอาหาร หรือไวรัส ควรตรวจสอบดินและใส่ปุ๋ยบำรุง")

# Footer (อยู่นอกกรอบ)
st.markdown("""
    <div style="text-align: center; margin-top: 40px; color: rgba(255,255,255,0.8); font-size: 0.8rem;">
        โครงงานวิจัยทางคอมพิวเตอร์ • มหาวิทยาลัยราชภัฏอุบลราชธานี<br>
        พัฒนาโดย: แมวสีขาวเทา และผองเพื่อน
    </div>
""", unsafe_allow_html=True)
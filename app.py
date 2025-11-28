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

# --- 2. 🎨 CSS ตกแต่ง (Gray Box Theme) ---
def local_css():
    st.markdown("""
    <style>
        /* นำเข้าฟอนต์ Prompt */
        @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
        
        /* บังคับฟอนต์ทั้งหน้า */
        html, body, [class*="css"] {
            font-family: 'Prompt', sans-serif;
        }
        
        /* 1. พื้นหลังหลัก (Background) - เปลี่ยนเป็นสีเข้มเพื่อให้กล่องเทาเด่นขึ้น */
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #222222 !important; /* พื้นหลังเว็บสีดำเทา */
            background-image: none !important;
        }

        /* 2. ปรับแต่ง "กรอบ/การ์ด" (เปลี่ยนจาก Glass เป็น Gray Box ตามที่ขอ) */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #e0e0e0 !important;       /* พื้นหลังสีเทา */
            border: 2px solid #333333 !important;       /* ขอบสีเข้มและหนาขึ้น */
            box-shadow: 4px 4px 10px rgba(0,0,0,0.5) !important; /* ใส่เงาชัดๆ */
            border-radius: 10px !important;             /* มุมมน */
            
            padding: 40px 30px !important;
            margin-bottom: 20px;
            
            /* Animation ตอนเปิดเว็บ */
            animation: fadeUp 0.8s ease-out;
        }
        
        /* ป้องกันสีพื้นหลังซ้อนทับ */
        [data-testid="stVerticalBlockBorderWrapper"] > div {
            background-color: transparent !important;
        }
        
        /* ซ่อน Header/Footer เดิม */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}

        /* 3. ส่วนหัว (Icon & Titles) */
        .card-header-custom {
            text-align: center;
            margin-bottom: 20px;
        }
        
        .app-icon {
            width: 100px;
            height: 100px;
            background-color: #ffffff; /* พื้นหลังไอคอนสีขาว */
            border: 2px solid #333;    /* ขอบไอคอนสีดำ */
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 50px;
            margin: 0 auto 20px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
            cursor: default;
        }
        
        .subtitle {
            color: #d32f2f;
            font-weight: 700;
            font-size: 0.9rem;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        
        h1 {
            color: #000000 !important; /* หัวข้อสีดำสนิท */
            font-weight: 800 !important;
            font-size: 2rem !important;
            margin: 0 !important;
            padding: 0 !important;
            text-align: center;
            text-transform: uppercase;
        }
        
        /* 4. คำอธิบาย (Description) */
        .description {
            color: #333333; /* ตัวหนังสือคำอธิบายสีเข้ม */
            font-weight: 500;
            font-size: 0.95rem;
            line-height: 1.6;
            text-align: center;
            margin-bottom: 30px;
        }

        /* 5. ปุ่มกด (Button) - ปรับสไตล์ให้เข้ากับธีม */
        div.stButton > button {
            background-color: #333333 !important; /* ปุ่มสีดำ */
            background-image: none !important;
            color: white !important;
            border: 2px solid #000 !important;
            border-radius: 50px !important;
            padding: 12px 40px !important;
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
            width: 100% !important;
            transition: all 0.2s ease !important;
        }
        div.stButton > button:hover {
            transform: scale(1.02) !important;
            background-color: #555555 !important; /* ชี้แล้วเปลี่ยนสี */
            color: white !important;
        }
        
        /* 6. File Uploader */
        [data-testid="stFileUploaderDropzone"] {
            background-color: #ffffff !important; /* พื้นที่อัปโหลดสีขาว */
            border: 2px dashed #333333 !important; /* เส้นปะสีดำ */
            border-radius: 15px !important;
            color: #333 !important;
        }
        [data-testid="stFileUploaderDropzone"] div {
            color: #333 !important;
        }
        
        /* 7. Footer */
        .footer-credit {
            font-size: 0.8rem;
            color: #555;
            margin-top: 30px;
            padding-top: 15px;
            text-align: center;
            border-top: 2px solid #ccc;
            font-weight: 500;
        }
        .badge-custom {
            background-color: #333;
            color: #fff;
            padding: 0.35em 0.65em;
            font-size: 0.75em;
            font-weight: 700;
            border-radius: 0.25rem;
            display: inline-block;
            margin-top: 10px;
        }

        /* Animation Keyframes */
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(40px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """, unsafe_allow_html=True)
local_css()

# --- 3. โหลดโมเดล ---
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

# --- ⭐ สร้างกรอบสไตล์ Gray Box (ตามคำขอ) ⭐ ---
with st.container(border=True):
    
    # 1. ส่วนหัว (Icon + Titles)
    st.markdown("""
        <div class="card-header-custom">
            <div class="app-icon">🌶️</div>
            <div class="subtitle">AI Expert System</div>
            <h1>Chili Doctor AI</h1>
        </div>
        
        <p class="description">
            ระบบผู้เชี่ยวชาญปัญญาประดิษฐ์สำหรับวินิจฉัยโรคพริกจากใบ <br>
            ด้วยเทคโนโลยี <strong>Deep Learning (EfficientNetB4)</strong> <br>
            ความแม่นยำสูง รวดเร็ว และใช้งานง่าย
        </p>
    """, unsafe_allow_html=True)

    # 2. ส่วนอัปโหลด
    file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    if file is None:
        st.markdown("""
            <div style="text-align: center; margin-top: 10px;">
                <small style="color: #555; font-weight: bold;">*แนะนำให้เปิดผ่าน Google Chrome หรือ Safari</small>
            </div>
        """, unsafe_allow_html=True)

    # 3. ส่วนแสดงผล
    if file is not None:
        image = Image.open(file)
        
        st.markdown("<br>", unsafe_allow_html=True)
        # จัดรูปให้อยู่ตรงกลางสวยๆ
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, use_container_width=True)
            
        if st.button("🚀 เริ่มต้นวิเคราะห์โรค"):
            if model is None:
                st.error("❌ ไม่สามารถโหลดโมเดลได้")
            else:
                with st.spinner('AI กำลังประมวลผล...'):
                    predictions = import_and_predict(image, model)
                    class_names = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellow']
                    class_index = np.argmax(predictions)
                    result_class = class_names[class_index]
                    confidence = np.max(predictions) * 100

                # เส้นคั่นภายในการ์ด
                st.markdown("<hr style='margin: 30px 0; border-top: 2px solid #bbb;'>", unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div style="text-align: center;">
                        <h3 style="color: #333; margin-bottom: 5px;">ผลการวิเคราะห์</h3>
                        <h1 style="color: #d32f2f; font-size: 2.5rem; margin: 0; text-shadow: 1px 1px 0px white;">{result_class}</h1>
                        <p style="color: #333; font-weight: bold;">ความมั่นใจ: <b>{confidence:.2f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)

                # --- จัดการ Icon และคำแนะนำตามโรค ---
                treatment_text = ""
                bg_color = "#fff3cd"
                text_color = "#856404"
                border_color = "#ffecb5"
                icon = "⚠️" # ไอคอนเริ่มต้น
                
                if result_class == 'healthy':
                    treatment_text = "ต้นพริกแข็งแรงดี! ไม่พบร่องรอยโรค หมั่นดูแลรดน้ำและใส่ปุ๋ยตามปกติ"
                    bg_color = "#d4edda"
                    text_color = "#155724"
                    border_color = "#c3e6cb"
                    icon = "🌿"
                elif result_class == 'leaf curl':
                    treatment_text = "โรคใบหงิกมักเกิดจากแมลงหวี่ขาว ให้กำจัดวัชพืชและใช้สารสกัดสะเดา หรือเชื้อราเมตาไรเซียมฉีดพ่น"
                    icon = "🌀"
                elif result_class == 'leaf spot':
                    treatment_text = "โรคใบจุดตากบ เกิดจากเชื้อรา ให้ตัดแต่งใบที่เป็นโรคเผาทำลาย และฉีดพ่นสารป้องกันเชื้อรา"
                    icon = "🍂"
                elif result_class == 'whitefly':
                      treatment_text = "พบแมลงหวี่ขาว ให้ใช้กับดักกาวเหนียวสีเหลือง หรือฉีดพ่นน้ำหมักสมุนไพร"
                      icon = "🪰"
                elif result_class == 'yellow':
                      treatment_text = "อาการใบเหลือง อาจเกิดจากการขาดสารอาหาร หรือไวรัส ควรตรวจสอบดินและใส่ปุ๋ยบำรุง"
                      icon = "🟡"
                
                # กล่องผลลัพธ์ (ให้ขอบเข้มขึ้นตามธีม)
                st.markdown(f"""
                    <div style="background-color: {bg_color}; color: {text_color}; border: 2px solid {text_color}; padding: 20px; border-radius: 12px; margin-top: 15px; font-size: 0.95rem; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                        <div style="display: flex; align-items: start;">
                            <div style="font-size: 1.8rem; margin-right: 15px;">{icon}</div>
                            <div>
                                <strong style="display: block; margin-bottom: 5px;">คำแนะนำ:</strong>
                                {treatment_text}
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    # 4. Footer (Credit)
    st.markdown("""
        <div class="footer-credit">
            โครงงานวิจัยทางคอมพิวเตอร์ <br>
            <strong>มหาวิทยาลัยราชภัฏอุบลราชธานี</strong> <br>
            <span class="badge-custom">v.1.0 (Final Release)</span> <br>
            <div style="margin-top: 10px; font-size: 0.75rem; color: #555;">
                พัฒนาโดย: แมวสีขาวเทา และผองเพื่อน
            </div>
        </div>
    """, unsafe_allow_html=True)
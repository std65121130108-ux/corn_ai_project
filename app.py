import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import time
import os
import mysql.connector
import io
import gdown
import urllib.parse
import requests

# --- ฟังก์ชันช่วย Rerun (แก้ปัญหา AttributeError) ---
def force_rerun():
    if hasattr(st, 'rerun'):
        st.rerun()
    else:
        st.experimental_rerun()

# --- [ส่วนสำคัญ 1] สร้างไฟล์ Config บังคับ Light Mode ---
config_dir = ".streamlit"
config_path = os.path.join(config_dir, "config.toml")

if not os.path.exists(config_dir):
    os.makedirs(config_dir)

with open(config_path, "w") as f:
    f.write('[theme]\nbase="light"\nprimaryColor="#F9A825"\nbackgroundColor="#FFFFFF"\nsecondaryBackgroundColor="#FFF8E1"\ntextColor="#333333"\n')

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="Corn Doctor AI",
    page_icon="🌽",
    layout="centered"
)

# --- 2. CSS ตกแต่ง ---
def local_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600;800&display=swap');
        
        html, body, [class*="css"], [data-testid="stAppViewContainer"] {
            font-family: 'Prompt', sans-serif !important;
            color: #333333 !important;
            font-weight: 400 !important;
        }

        .stApp {
            background: linear-gradient(135deg, #a8ff78 0%, #78ffd6 100%) !important;
            background-attachment: fixed !important;
            background-size: cover !important;
        }

        header[data-testid="stHeader"] { background-color: transparent !important; }
        div[data-testid="stDecoration"] { display: none; }

        ul[data-testid="main-menu-list"] {
            background-color: #ffffff !important;
            border-radius: 10px !important;
            padding: 10px !important;
        }
        ul[data-testid="main-menu-list"] * {
            color: #333333 !important;
            background-color: #ffffff !important;
        }
        button[kind="header"] { color: #ffffff !important; }

        div.block-container {
            background-color: rgba(255, 255, 255, 0.95) !important;
            border-radius: 30px !important;
            padding: 2rem 2rem 4rem 2rem !important; 
            margin-top: 2rem !important;
            box-shadow: 0 15px 50px rgba(0,0,0,0.3) !important;
            min-height: auto !important;
        }

        .app-header-icon {
            font-size: 80px !important;
            background: radial-gradient(circle, #fff176 0%, #fbc02d 100%) !important;
            width: 140px; height: 140px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 15px auto !important;
            box-shadow: 0 10px 25px rgba(255, 193, 7, 0.4) !important;
            border: 5px solid #ffffff !important;
        }

        div[role="radiogroup"] {
            display: flex !important;
            flex-direction: row !important;
            gap: 10px !important;
            justify-content: center !important;
            flex-wrap: wrap !important;
        }
        div[role="radiogroup"] label {
            background: linear-gradient(135deg, #fbc02d 0%, #f57f17 100%) !important;
            border: none !important;
            padding: 10px 20px !important;
            border-radius: 25px !important;
            cursor: pointer !important;
            transition: all 0.2s !important;
            margin: 0 !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important;
            color: #ffffff !important; 
        }
        div[role="radiogroup"] label p {
            color: #ffffff !important;
            font-weight: 400 !important;
            font-size: 1rem !important;
        }
        div[role="radiogroup"] label:hover {
            filter: brightness(1.1) !important;
            transform: translateY(-2px) !important;
        }
        .stRadio > label {
            color: #e65100 !important;
            font-weight: 800 !important;
            font-size: 1.3rem !important;
            margin-bottom: 15px !important;
            display: block;
            text-align: center;
        }

        div.stButton > button {
            background: linear-gradient(135deg, #fbc02d 0%, #f57f17 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 15px !important;
            font-weight: 400 !important;
            transition: all 0.3s ease !important;
            width: 100% !important;
            padding: 0.8rem !important;
            font-size: 1rem !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
        }
        div.stButton > button:hover {
            filter: brightness(1.1) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 15px rgba(0,0,0,0.3) !important;
            color: #ffffff !important;
        }
        div[data-testid="column"] button {
             background: linear-gradient(135deg, #fbc02d 0%, #f57f17 100%) !important;
             color: #ffffff !important;
             border: none !important;
        }

        div[data-testid="stImage"] > img {
            border-radius: 20px;
            max-height: 350px;
            width: auto;
            max-width: 100%;
            margin: 0 auto;
            display: block;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .footer-credit {
            font-size: 0.8rem; color: #888; text-align: center; margin-top: 10px;
        }
        
        h1 { 
            text-align: center; color: #e65100 !important; 
            font-weight: 800 !important; font-size: 2.2rem !important;
            margin-bottom: 5px !important; text-shadow: 2px 2px 0px #fff8e1;
        }

        .custom-home-btn {
            background: linear-gradient(135deg, #fbc02d 0%, #f57f17 100%);
            color: #ffffff !important;
            text-decoration: none;
            padding: 0.8rem 2rem;
            border-radius: 15px;
            font-weight: 400;
            font-family: 'Prompt', sans-serif;
            display: inline-block;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
            text-align: center;
            width: 100%;
            border: none;
        }
        .custom-home-btn:hover {
            filter: brightness(1.1);
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(0,0,0,0.3);
            color: #ffffff !important;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- 3. ฟังก์ชัน Database ---
BASE_IMAGE_URL = "http://www.cedubru.com/corn/uploads/" 

def init_connection():
    return mysql.connector.connect(
        host="www.cedubru.com",
        user="cedubruc_corn_db_s",      
        password="bcbbDrypgCQXnSYu8Qrw", 
        database="cedubruc_corn_db_s"   
    )

def get_image_list(filter_mode):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        
        base_sql = """
            SELECT p.case_id, m.file_path, p.ai_prediction 
            FROM plant_cases p
            JOIN media_files m ON p.case_id = m.case_id
            WHERE m.file_type = 'image'
        """
        
        if "ยังไม่ตรวจ" in filter_mode:
            sql = base_sql + " AND p.status = 'NEW' ORDER BY p.case_id ASC"
        elif "ตรวจแล้ว" in filter_mode:
            sql = base_sql + " AND p.status IN ('AI_ANALYZED', 'EXPERT_CONFIRMED') ORDER BY p.case_id DESC"
        else:
            sql = base_sql + " ORDER BY p.case_id DESC"
            
        cursor.execute(sql)
        data = cursor.fetchall()
        conn.close()
        return data
    except Exception as e:
        st.error(f"❌ DB Error: {e}")
        return []

def get_image_data(file_path, case_id):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT ai_prediction, ai_confidence FROM plant_cases WHERE case_id = %s", (case_id,))
        result_data = cursor.fetchone()
        conn.close()
        
        saved_result = result_data[0] if result_data else None
        saved_conf = result_data[1] if result_data else 0

        img_url = BASE_IMAGE_URL + file_path
        try:
            response = requests.get(img_url, timeout=10)
            if response.status_code == 200:
                image_bytes = response.content
                return image_bytes, saved_result, saved_conf
            else:
                st.warning(f"โหลดรูปไม่ได้ (HTTP {response.status_code}): {img_url}")
                return None
        except:
            return None
    except:
        return None

def update_database(case_id, result, confidence):
    try:
        conn = init_connection()
        cursor = conn.cursor()
        if result is None:
            sql = "UPDATE plant_cases SET ai_prediction=NULL, ai_confidence=0, status='NEW' WHERE case_id=%s"
            cursor.execute(sql, (case_id,))
        else:
            sql = "UPDATE plant_cases SET ai_prediction=%s, ai_confidence=%s, status = IF(status='EXPERT_CONFIRMED', status, 'AI_ANALYZED') WHERE case_id=%s"
            cursor.execute(sql, (result, float(confidence), case_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Update Error: {e}")
        return False

# --- 4. Load Model ---
if hasattr(st, 'cache_resource'): cache_decorator = st.cache_resource
else: cache_decorator = st.experimental_singleton

@cache_decorator
def load_model():
    filename = 'corn_model_full_v1.h5'
    file_id = '1qWALZiNUsohslr5ADgOESE3YRS5P_OS_'
    url = f'https://drive.google.com/uc?id={file_id}'

    if not os.path.exists(filename):
        with st.spinner("⏳ กำลังดาวน์โหลดโมเดลข้าวโพด... (ครั้งแรกอาจนานหน่อย)"):
            try:
                gdown.download(url, filename, quiet=False)
            except Exception as e:
                st.error(f"❌ ดาวน์โหลดไม่สำเร็จ: {e}")
                return None

    try:
        return tf.keras.models.load_model(filename)
    except Exception as e:
        st.error(f"❌ โมเดลเสียหาย: {e}")
        return None

def import_and_predict(image_data, model):
    size = (380, 380)
    try:
        image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    except AttributeError:
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img_array = np.asarray(image).astype(np.float32)
    data = np.ndarray(shape=(1, 380, 380, 3), dtype=np.float32)
    data[0] = img_array
    return model.predict(data)

# --- 5. Main UI ---
model = load_model()

st.markdown("""
    <div class='app-header-icon'>🌽</div>
    <h1>Corn Doctor AI</h1>
    <p style='text-align: center; color: #555; margin-bottom: 30px; font-size: 1.1rem;'>
        ระบบวินิจฉัยโรคใบข้าวโพดอัจฉริยะด้วย AI
    </p>
""", unsafe_allow_html=True)

# --- ตัวกรอง ---
c1, c2, c3 = st.columns([0.1, 3, 0.1])
with c2:
    filter_option = st.radio(
        "📂 ตัวกรองข้อมูล:", 
        ["ทั้งหมด (All)", "ตรวจแล้ว (Analyzed)", "ยังไม่ตรวจ (Pending)"], 
        index=0 
    )

# ดึงข้อมูล
image_list = get_image_list(filter_option)

if len(image_list) > 0:
    id_list = [row[0] for row in image_list] 
    
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if st.session_state.current_index >= len(id_list):
        st.session_state.current_index = 0

    current_idx = st.session_state.current_index
    current_case_id = image_list[current_idx][0]
    current_file_path = image_list[current_idx][1]
    
    # --- แสดงผล ---
    st.markdown("---")
    st.markdown(f"<div style='text-align: center; color: #333; margin-bottom: 15px; font-weight: normal; font-size: 1.1rem; background: #fff8e1; padding: 10px; border-radius: 10px;'>📸 รูปที่ {current_idx + 1} / {len(id_list)} (Case ID: {current_case_id})</div>", unsafe_allow_html=True)

    data_row = get_image_data(current_file_path, current_case_id)
    
    if data_row:
        blob_data, saved_result, saved_conf = data_row
        image = Image.open(io.BytesIO(blob_data))
        
        col_img, col_act = st.columns([1, 1])
        
        with col_img:
            st.image(image, use_column_width=True)
            st.caption(f"File: {current_file_path}")
        
        with col_act:
            st.markdown("### ผลการวิเคราะห์")
            
            if saved_result:
                bg = "#d4edda" if 'Healthy' in saved_result or 'ปกติ' in saved_result else "#f8d7da"
                text_col = "#155724" if 'Healthy' in saved_result or 'ปกติ' in saved_result else "#721c24"
                
                st.markdown(f"""
                    <div style="background-color: {bg}; padding: 20px; border-radius: 15px; border: 2px solid {text_col}; margin-bottom: 20px; text-align: center;">
                        <h2 style="color: {text_col} !important; margin: 0; font-size: 1.6rem; font-weight: 400;">{saved_result}</h2>
                        <p style="margin-top: 10px; font-size: 1rem; color: #333;">ความมั่นใจ: <strong>{saved_conf:.2f}%</strong></p>
                    </div>
                """, unsafe_allow_html=True)
                
                if st.button("🔄 วินิจฉัยซ้ำ"):
                    update_database(current_case_id, None, 0)
                    force_rerun()
            
            else:
                st.info("⚠️ รูปนี้ยังไม่ได้รับการตรวจสอบ")
                # ปุ่มวิเคราะห์เดี่ยว (แก้ไข: ให้แสดงผลลัพธ์ก่อนรีโหลด)
                if st.button("🚀 วินิจฉัยรูปนี้"):
                    if model:
                        with st.spinner("AI กำลังทำงาน..."):
                            preds = import_and_predict(image, model)
                            
                            class_names = ['Common_Rust', 'Gray_Leaf_Spot', 'Blight', 'Healthy']
                            
                            idx = np.argmax(preds)
                            res_eng = class_names[idx]
                            conf = np.max(preds) * 100
                            
                            th_dict = {
                                'Common_Rust': 'โรคราสนิม (Common Rust)',
                                'Gray_Leaf_Spot': 'โรคใบจุดสีเทา (Gray Leaf Spot)',
                                'Blight': 'โรคใบไหม้แผลใหญ่ (Blight)',
                                'Healthy': 'ปกติ (Healthy)'
                            }
                            final_res = th_dict.get(res_eng, res_eng)
                            
                            update_database(current_case_id, final_res, conf)

                            # --- ส่วนแสดงผลลัพธ์ทันที ---
                            bg = "#d4edda" if 'Healthy' in final_res or 'ปกติ' in final_res else "#f8d7da"
                            text_col = "#155724" if 'Healthy' in final_res or 'ปกติ' in final_res else "#721c24"

                            st.markdown(f"""
                                <div style="background-color: {bg}; padding: 20px; border-radius: 15px; border: 2px solid {text_col}; margin-top: 20px; text-align: center;">
                                    <h3 style="color: {text_col} !important; margin: 0;">ผลการวินิจฉัยล่าสุด</h3>
                                    <h2 style="color: {text_col} !important; margin: 10px 0; font-size: 1.8rem; font-weight: 800;">{final_res}</h2>
                                    <p style="margin-top: 10px; font-size: 1rem; color: #333;">ความมั่นใจ: <strong>{conf:.2f}%</strong></p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            st.success("✅ บันทึกเรียบร้อย! (กำลังรีโหลดหน้าเว็บ...)")
                            time.sleep(3) # รอให้ผู้ใช้อ่านผลลัพธ์ 3 วินาที
                            force_rerun()
                    else:
                        st.error("โมเดลยังไม่โหลด")
                
                # --- ปุ่ม Batch Scan ---
                if "ยังไม่ตรวจ" in filter_option:
                    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                    if st.button(f"⚡ วิเคราะห์ทั้งหมดที่เหลือ ({len(image_list)} รูป)"):
                        if model:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            for i, (c_id, f_path, _) in enumerate(image_list):
                                status_text.text(f"⏳ กำลังวิเคราะห์... {i+1}/{len(image_list)}")
                                d_row = get_image_data(f_path, c_id)
                                if d_row:
                                    img_b = d_row[0]
                                    img_pil = Image.open(io.BytesIO(img_b))
                                    p = import_and_predict(img_pil, model)
                                    
                                    class_names = ['Common_Rust', 'Gray_Leaf_Spot', 'Blight', 'Healthy']
                                    idx = np.argmax(p)
                                    res_eng = class_names[idx]
                                    conf = np.max(p) * 100
                                    
                                    th_dict = {
                                        'Common_Rust': 'โรคราสนิม (Common Rust)',
                                        'Gray_Leaf_Spot': 'โรคใบจุดสีเทา (Gray Leaf Spot)',
                                        'Blight': 'โรคใบไหม้แผลใหญ่ (Blight)',
                                        'Healthy': 'ปกติ (Healthy)'
                                    }
                                    final_res = th_dict.get(res_eng, res_eng)
                                    update_database(c_id, final_res, conf)
                                
                                progress_bar.progress((i + 1) / len(image_list))
                            
                            status_text.text("✅ เสร็จสิ้น!")
                            st.success("บันทึกข้อมูลเรียบร้อยแล้ว")
                            time.sleep(1)
                            force_rerun()

    # --- ปุ่มนำทาง ---
    st.markdown("<br>", unsafe_allow_html=True) 
    c_prev, c_empty, c_next = st.columns([1, 0.2, 1]) 
    
    with c_prev:
        is_first_image = st.session_state.current_index == 0
        if is_first_image:
            if st.button("⏮️ ไปรูปสุดท้าย"):
                st.session_state.current_index = len(id_list) - 1
                force_rerun()
        else:
            if st.button("◀️ ย้อนกลับ"):
                st.session_state.current_index -= 1
                force_rerun()
            
    with c_next:
        is_last_image = st.session_state.current_index >= len(id_list) - 1
        if is_last_image:
            if st.button("🔄 เริ่มต้นใหม่"):
                st.session_state.current_index = 0
                force_rerun()
        else:
            if st.button("ถัดไป ▶️"):
                st.session_state.current_index += 1
                force_rerun()

else:
    st.warning("ไม่พบข้อมูลในฐานข้อมูลตามตัวกรองที่เลือก")

# --- ปุ่มลิงก์ HTML ---
base_url = "http://www.cedubru.com/"
path = "corn/" 
full_url = base_url + path

st.markdown(f"""
    <div style="text-align: center; margin-top: 30px; margin-bottom: 20px;">
        <a href="{full_url}" target="_blank" class="custom-home-btn">
            🏠 คลิกเพื่อกลับสู่หน้าหลัก
        </a>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="footer-credit">
        <strong>ระบบวินิจฉัยโรคใบข้าวโพด V.1.0</strong>
    </div>
""", unsafe_allow_html=True)
import streamlit as st          # ไลบรารีหลักสำหรับสร้าง Web App
import tensorflow as tf         # ไลบรารี AI สำหรับรันโมเดล Deep Learning
from PIL import Image, ImageOps # เครื่องมือจัดการรูปภาพ (ย่อ/ขยาย/ตัด)
import numpy as np              # เครื่องมือคำนวณตัวเลขและอาเรย์ (Matrix)
import time                     # เครื่องมือเกี่ยวกับเวลา (ใช้หน่วงเวลา/จับเวลา)
import os                       # เครื่องมือจัดการไฟล์ในเครื่อง (เช็คว่ามีไฟล์ไหม)

# --- 1. ตั้งค่าหน้าเว็บ (Page Config) ---
# บรรทัดนี้ต้องอยู่บนสุดเสมอ เป็นการบอก Browser ว่าเว็บเราชื่ออะไร ไอคอนอะไร
st.set_page_config(
    page_title="Chili Doctor AI",
    page_icon="🌶️",
    layout="centered" # จัดหน้าเว็บให้อยู่กึ่งกลาง (เหมาะกับมือถือ)
)

# --- 2. 🎨 CSS ตกแต่ง (Custom CSS) ---
# ฟังก์ชันนี้จะ "ฉีด" โค้ด HTML/CSS เข้าไปเพื่อเปลี่ยนหน้าตาเว็บให้สวยงาม
def local_css():
    st.markdown("""
    <style>
        /* นำเข้าฟอนต์ 'Prompt' จาก Google Fonts เพื่อให้ภาษาไทยสวยงาม */
        @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
        
        /* 1. Global Settings: บังคับให้ทุกส่วนใช้ฟอนต์ Prompt */
        html, body, [class*="css"] {
            font-family: 'Prompt', sans-serif;
        }

        /* 2. พื้นหลัง Gradient: ไล่สีส้มแดงให้ดูร้อนแรงเหมือนพริก */
        .stApp {
            background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%) !important;
            background-attachment: fixed !important;
            background-size: cover !important;
        }

        /* 3. Block Container: คือการ์ดสีขาวตรงกลางหน้าจอ */
        div.block-container {
            background-color: rgba(255, 255, 255, 0.95) !important; /* พื้นหลังสีขาวโปร่งแสงนิดๆ */
            border-radius: 25px !important;    /* มุมโค้งมน */
            padding: 3rem 2rem !important;     /* ระยะห่างขอบ */
            margin-top: 2rem !important;       /* ดันลงมาจากด้านบน */
            box-shadow: 0 10px 40px rgba(0,0,0,0.3) !important; /* เงาให้ดูลอยเด่น */
            max-width: 700px !important;       /* กำหนดความกว้างสูงสุด */
        }

        /* บังคับให้ตัวหนังสือในกล่องขาวเป็นสีเข้ม (จะได้อ่านง่าย) */
        div.block-container h1, div.block-container h2, div.block-container h3, 
        div.block-container p, div.block-container span, div.block-container div, 
        div.block-container label, div.block-container small {
             color: #333333 !important;
        }
        
        /* ยกเว้น Text ในปุ่มกด ให้เป็นสีขาวเหมือนเดิม */
        div.stButton > button p { color: white !important; }

        /* ซ่อน Header/Footer ดั้งเดิมของ Streamlit (ขีด 3 ขีดขวาบน และเครดิตล่างสุด) */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}

        /* 4. Custom Elements: ตกแต่งโลโก้แอป */
        .app-icon {
            width: 100px;
            height: 100px;
            background: linear-gradient(45deg, #ff9a9e 0%, #fad0c4 99%, #fad0c4 100%) !important;
            border-radius: 50%;   /* ทำเป็นวงกลม */
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 50px;
            margin: 0 auto 20px;  /* จัดกึ่งกลาง */
            box-shadow: 0 6px 20px rgba(255, 75, 43, 0.4) !important;
            cursor: default;
        }
        
        /* ตกแต่งหัวข้อย่อย */
        .subtitle {
            color: #d32f2f !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            letter-spacing: 2px !important;
            text-transform: uppercase !important;
            text-align: center !important;
            margin-bottom: 5px !important;
        }
        
        /* ตกแต่งหัวข้อใหญ่ (H1) */
        h1 {
            font-weight: 800 !important;
            font-size: 2.2rem !important;
            margin: 0 !important;
            padding: 0 !important;
            text-align: center !important;
        }

        /* ตกแต่งคำอธิบายแอป */
        .description {
            font-size: 1rem !important;
            line-height: 1.6 !important;
            text-align: center !important;
            margin: 20px 0 30px 0 !important;
        }

        /* 5. ปุ่มกด (Button Styling) */
        div.stButton > button {
            background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%) !important;
            border: none !important;
            color: white !important;
            padding: 15px 40px !important;
            border-radius: 50px !important; /* ปุ่มทรงแคปซูล */
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            box-shadow: 0 5px 15px rgba(255, 65, 108, 0.4) !important;
            width: 100% !important;
            transition: all 0.3s ease !important;
        }
        /* เอฟเฟกต์ตอนเอาเมาส์ชี้ปุ่ม */
        div.stButton > button:hover {
            transform: scale(1.02) !important; /* ขยายใหญ่นิดนึง */
            box-shadow: 0 8px 25px rgba(255, 65, 108, 0.5) !important;
        }

        /* 6. พื้นที่อัปโหลดไฟล์ (Dropzone) */
        [data-testid="stFileUploaderDropzone"] {
            background-color: rgba(240, 240, 240, 0.5) !important;
            border: 2px dashed #FF4B2B !important; /* เส้นประสีส้ม */
            border-radius: 15px !important;
            padding: 20px !important;
        }
        /* ปุ่ม Browse files */
        [data-testid="stFileUploaderDropzone"] button {
             border: none !important;
             background: #FF4B2B !important;
             color: white !important;
        }

        /* 7. Tabs Styling (แท็บเลือกถ่ายรูป/อัปโหลด) */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: transparent;
            margin-bottom: 20px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 45px;
            background-color: #f0f0f0;
            border-radius: 20px;
            padding: 0px 20px;
            color: #666;
            font-weight: 600;
            border: none;
        }
        /* สีตอนเลือกแท็บนั้นอยู่ */
        .stTabs [aria-selected="true"] {
            background-color: #ffe5e5 !important;
            color: #FF4B2B !important;
            border: 1px solid #FF4B2B !important;
        }

        /* 8. Footer (ส่วนเครดิตด้านล่าง) */
        .footer-credit {
            font-size: 0.8rem !important;
            color: #888 !important;
            margin-top: 30px !important;
            padding-top: 20px !important;
            text-align: center !important;
            border-top: 1px solid rgba(0,0,0,0.1) !important;
        }
        .badge-custom {
            background-color: #f0f0f0 !important;
            color: #333 !important;
            padding: 0.35em 0.8em !important;
            font-size: 0.75em !important;
            font-weight: 700 !important;
            border-radius: 20px !important;
            display: inline-block !important;
            margin-top: 10px !important;
        }
    </style>
    """, unsafe_allow_html=True)

# เรียกใช้ฟังก์ชัน CSS ทันทีที่รัน
local_css()

# --- 3. โหลดโมเดล AI ---
# ใช้ @st.experimental_singleton เพื่อ Cache โมเดลไว้ในแรม
# ทำให้ไม่ต้องโหลดใหม่ทุกครั้งที่ผู้ใช้กดปุ่ม (ช่วยให้เว็บเร็วขึ้นมาก)
# สร้าง Logic เช็คเวอร์ชัน (ถ้าใหม่ใช้ cache_resource ถ้าเก่าใช้ singleton)
if hasattr(st, 'cache_resource'):
    cache_decorator = st.cache_resource
else:
    cache_decorator = st.experimental_singleton

@cache_decorator
def load_model():
    filename = 'chilli_efficientnetb4_full.h5'
    
    # เช็คว่ามีไฟล์โมเดลในเครื่องหรือยัง ถ้าไม่มีให้โหลดจาก Google Drive
    if not os.path.exists(filename):
        # ID ของไฟล์ Google Drive (ต้องเป็น Public Link)
        file_id = '1QJIaS61jMxx4vZ8uIVVz_IuGI6XAchLT' 
        url = f'https://drive.google.com/uc?id={file_id}'
        
        # สร้างพื้นที่ว่างไว้แสดงสถานะการดาวน์โหลด
        download_placeholder = st.empty()
        
        with download_placeholder.container():
            st.warning("""
                ⚠️ **กำลังดาวน์โหลดโมเดล AI (ครั้งแรกเท่านั้น)...**
                
                ไฟล์มีขนาดใหญ่ กรุณารอสักครู่ ระบบกำลังเตรียมความพร้อม...
            """)
            # แสดงวงกลมหมุนๆ ระหว่างโหลด
            with st.spinner("🚀 กำลังดึงข้อมูลจาก Server... (ห้ามปิดหน้านี้)"):
                try:
                    # ใช้ gdown เพื่อโหลดไฟล์จาก Drive
                    import gdown
                    gdown.download(url, filename, quiet=False)
                    
                    if os.path.exists(filename):
                        # โหลดเสร็จ แจ้งเตือนสีเขียว
                        download_placeholder.success("✅ ดาวน์โหลดเสร็จสิ้น! พร้อมใช้งาน")
                        time.sleep(2) # โชว์ค้างไว้ 2 วิ
                        download_placeholder.empty() # ลบข้อความทิ้ง
                    else:
                        download_placeholder.error("❌ ดาวน์โหลดไม่สำเร็จ กรุณาลองใหม่")
                        return None
                except Exception as e:
                    download_placeholder.error(f"❌ เกิดข้อผิดพลาด: {e}")
                    return None
                    
    # คำสั่งหลัก: โหลดโมเดลเข้าสู่ TensorFlow
    try:
        return tf.keras.models.load_model(filename)
    except:
        return None

# --- ฟังก์ชันทำนายผล (Prediction Function) ---
# รับรูปภาพเข้ามา -> แปลงเป็นตัวเลข -> ส่งให้ AI ทำนาย
def import_and_predict(image_data, model):
    size = (380, 380) # ขนาดที่ EfficientNetB4 ต้องการ
    
    # ปรับขนาดภาพ (Resize) และใช้ Filter คุณภาพสูง (LANCZOS)
    try:
        image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    except AttributeError:
        # เผื่อกรณีเซิร์ฟเวอร์ใช้ Pillow เวอร์ชันเก่า
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        
    # แปลงรูปเป็นอาเรย์ตัวเลข (Numpy Array)
    img_array = np.asarray(image).astype(np.float32)
    
    # เพิ่มมิติข้อมูลจาก (380,380,3) เป็น (1,380,380,3) เพราะ AI รับข้อมูลเป็น Batch
    data = np.ndarray(shape=(1, 380, 380, 3), dtype=np.float32)
    data[0] = img_array
    
    # สั่งให้โมเดลทำนาย
    return model.predict(data)

# --- 4. ส่วนแสดงผลหลัก (Main UI) ---

# เรียกใช้ฟังก์ชันโหลดโมเดล
model = load_model()

# ⭐ Session State: ระบบความจำของ Streamlit ⭐
# ใช้จำค่าต่างๆ ข้ามการรีโหลดหน้าเว็บ

# 1. ตัวนับการรีเซ็ต (ใช้เปลี่ยน Key ของ widget เพื่อบังคับล้างค่า)
if 'reset_count' not in st.session_state:
    st.session_state['reset_count'] = 0

# 2. ตัวจำภาพถ่าย (ถ้ามีภาพอยู่ จะซ่อนกล้องแล้วโชว์ภาพแทน)
if 'cam_img_buffer' not in st.session_state:
    st.session_state['cam_img_buffer'] = None

# --- ส่วนหัว (Header HTML) ---
st.markdown("""
    <div style="text-align: center;">
        <div class="app-icon">🌶️</div>
        <div class="subtitle">AI Expert System</div>
        <h1>Chili Doctor AI</h1>
        <p class="description">
            ระบบผู้เชี่ยวชาญปัญญาประดิษฐ์สำหรับวินิจฉัยโรคพริกจากใบ <br>
            ด้วยเทคโนโลยี <strong>Deep Learning (EfficientNetB4)</strong> <br>
            ความแม่นยำสูง รวดเร็ว และใช้งานง่าย
        </p>
    </div>
""", unsafe_allow_html=True)

# สร้างแท็บ 2 อัน: ถ่ายรูป / อัปโหลด
tab_cam, tab_up = st.tabs(["📸 ถ่ายภาพใบพริก", "📂 อัปโหลดไฟล์รูป"])

img_file_buffer = None
# สร้าง Key ให้ไม่ซ้ำกันตามจำนวนครั้งที่รีเซ็ต
camera_key = f"camera_{st.session_state['reset_count']}"
uploader_key = f"uploader_{st.session_state['reset_count']}"

# --- Logic Tab 1: กล้องถ่ายรูป ---
with tab_cam:
    # ถ้ายังไม่มีภาพในความจำ -> แสดงกล้องให้ถ่าย
    if st.session_state['cam_img_buffer'] is None:
        camera_image = st.camera_input("กล้องถ่ายรูป", key=camera_key)
        
        # แสดงป้ายแนะนำวิธีกดถ่าย
        st.markdown("""
            <div style="text-align: center; margin-top: 20px;">
                <div style="display: inline-block; background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%); padding: 15px 30px; border-radius: 50px; box-shadow: 0 5px 15px rgba(255, 65, 108, 0.4);">
                    <h4 style="color: #ffffff !important; margin: 0 !important; font-size: 1.1rem;">
                        📸 กดปุ่ม "Take Photo" ด้านบนเพื่อถ่ายรูป
                    </h4>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if camera_image is not None:
            # ถ้าผู้ใช้กดถ่ายรูป -> จำภาพลง Session State -> สั่งรีโหลดหน้าเว็บ (Rerun)
            st.session_state['cam_img_buffer'] = camera_image
            st.experimental_rerun()
            
    else:
        # ถ้ามีภาพในความจำแล้ว -> เอาภาพนั้นมาใช้เลย (ไม่ต้องโชว์กล้องซ้ำ)
        img_file_buffer = st.session_state['cam_img_buffer']
        st.success("✅ บันทึกภาพเรียบร้อยแล้ว (กดปุ่ม 'ถ่ายรูปใหม่อีกครั้ง' หากต้องการถ่ายรูปภาพใหม่)")

# --- Logic Tab 2: อัปโหลดไฟล์ ---
with tab_up:
    uploaded_file = st.file_uploader("เลือกรูปภาพจากเครื่อง", type=["jpg", "png", "jpeg"], key=uploader_key)
    if uploaded_file is not None:
        img_file_buffer = uploaded_file

# --- 5. แสดงผลรูปและปุ่มกด (Display & Actions) ---
if img_file_buffer is not None:
    # เปิดรูปด้วย PIL
    image = Image.open(img_file_buffer)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # จัด layout 3 คอลัมน์ เพื่อให้รูปอยู่ตรงกลางสวยๆ
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # ใส่กรอบให้รูปภาพ
        st.markdown('<div style="border-radius: 15px; overflow: hidden; box-shadow: 0 5px 15px rgba(0,0,0,0.1); border: 3px solid rgba(255,255,255,0.8);">', unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)

    # สร้างปุ่ม 2 ปุ่มเรียงกัน
    b1, b2 = st.columns(2, gap="medium")
    
    with b1:
        predict_click = st.button("🚀 วินิจฉัยโรคทันที")
        
    with b2:
        reset_click = st.button("🔄 ถ่ายรูปใหม่อีกครั้ง")

    # --- Logic ปุ่ม Reset ---
    if reset_click:
        st.session_state['reset_count'] += 1 # เพิ่มตัวนับเพื่อให้ Key เปลี่ยน (เป็นการบังคับล้างค่า)
        st.session_state['cam_img_buffer'] = None # ลบภาพในความจำ
        st.experimental_rerun() # รีโหลดหน้าเว็บใหม่

    # --- Logic ปุ่ม Predict (หัวใจสำคัญ) ---
    if predict_click:
        if model is None:
            st.error("❌ ไม่สามารถโหลดโมเดลได้")
        else:
            # หมุน Spinner รอระหว่างคำนวณ
            with st.spinner('AI กำลังประมวลผล...'):
                # ส่งรูปไปให้ AI ทำนาย
                predictions = import_and_predict(image, model)
                
                # รายชื่อโรค (ต้องเรียงตามลำดับเดียวกับตอนเทรนเป๊ะๆ)
                class_names = [
                    'Bacterial Spot', 
                    'Cercospora Leaf Spot', 
                    'Curl Virus', 
                    'Healthy Leaf', 
                    'Not leaf chilli', 
                    'Nutrition Deficiency', 
                    'White spot'
                ]
                
                # หาค่าที่มากที่สุด (argmax) ว่าตรงกับโรคไหน
                class_index = np.argmax(predictions)
                result_class = class_names[class_index]
                confidence = np.max(predictions) * 100 # แปลงเป็นเปอร์เซ็นต์

            # ขีดเส้นคั่น
            st.markdown("<div style='height: 1px; background-color: rgba(0,0,0,0.1); margin: 30px 0;'></div>", unsafe_allow_html=True)
            
            # แปลงชื่อภาษาอังกฤษ เป็นชื่อภาษาไทยสำหรับแสดงผล
            display_name = result_class
            if result_class == 'Bacterial Spot': display_name = "โรคจุดแบคทีเรีย (Bacterial Spot)"
            elif result_class == 'Cercospora Leaf Spot': display_name = "โรคใบจุดตากบ (Cercospora)"
            elif result_class == 'Curl Virus': display_name = "โรคใบหงิกไวรัส (Curl Virus)"
            elif result_class == 'Healthy Leaf': display_name = "ต้นพริกแข็งแรง (Healthy)"
            elif result_class == 'Not leaf chilli': display_name = "⚠️ ไม่ใช่รูปใบพริก"
            elif result_class == 'Nutrition Deficiency': display_name = "อาการขาดสารอาหาร (Deficiency)"
            elif result_class == 'White spot': display_name = "โรคจุดขาว (White Spot)"

            # แสดงผลลัพธ์ตัวใหญ่ๆ
            st.markdown(f"""
                <div style="text-align: center;">
                    <h3 style="color: #666; font-size: 1rem; margin-bottom: 5px;">ผลการวิเคราะห์</h3>
                    <h1 style="color: #FF4B2B !important; font-size: 2.2rem; margin: 0;">{display_name}</h1>
                    <div style="background: #fff0f0; color: #FF4B2B; display: inline-block; padding: 5px 15px; border-radius: 20px; font-weight: bold; font-size: 0.9rem; margin-top: 10px;">
                        ความมั่นใจ: {confidence:.2f}%
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # เตรียมข้อความคำแนะนำ และไอคอน ตามโรคที่เจอ
            treatment_text = ""
            bg_color = "rgba(255, 248, 225, 0.8)"
            icon_bg = "#ffecb3"
            icon = "⚠️"
            
            if result_class == 'Healthy Leaf':
                treatment_text = "ยอดเยี่ยม! ต้นพริกของคุณแข็งแรงดี ไม่พบร่องรอยโรค หมั่นรดน้ำและใส่ปุ๋ยตามปกติเพื่อรักษาผลผลิต"
                bg_color = "rgba(232, 245, 233, 0.8)" # สีเขียวอ่อน
                icon_bg = "#c8e6c9"
                icon = "🌿"
                
            elif result_class == 'Bacterial Spot':
                treatment_text = "โรคจุดแบคทีเรีย: ระบาดได้ดีในหน้าฝน ให้เก็บใบที่เป็นโรคไปเผาทำลาย และฉีดพ่นสารประกอบทองแดง (Copper) หรือใช้เชื้อแบคทีเรียบาซิลลัส (BS) ในการควบคุม"
                icon = "🟤"
                
            elif result_class == 'Cercospora Leaf Spot':
                treatment_text = "โรคใบจุดตากบ (เชื้อรา): มักเกิดจุดกลมสีน้ำตาล ให้ตัดแต่งใบที่ระบาดออก เพื่อให้อากาศถ่ายเท และฉีดพ่นสารป้องกันกำจัดเชื้อรากลุ่มแมนโคเซบ หรือคาร์เบนดาซิม"
                icon = "🍂"
                
            elif result_class == 'Curl Virus':
                treatment_text = "โรคใบหงิก (ไวรัส): เกิดจากแมลงพาหะ เช่น เพลี้ยไฟ/แมลงหวี่ขาว หากเป็นรุนแรงควรถอนทิ้งทันทีเพื่อป้องกันการลาม ป้องกันโดยการกำจัดแมลงพาหะอย่างสม่ำเสมอ"
                icon = "🌀"
                
            elif result_class == 'Nutrition Deficiency':
                treatment_text = "อาการขาดสารอาหาร: ใบอาจมีสีเหลืองซีด หรือเส้นใบเขียวแต่เนื้อใบเหลือง ควรปรับปรุงดิน ตรวจวัดค่า pH และเติมปุ๋ยธาตุอาหารรอง/เสริม (เช่น แมกนีเซียม, เหล็ก, แคลเซียม)"
                icon = "🟡"
            
            elif result_class == 'White spot':
                treatment_text = "โรคจุดขาว: อาจเกิดจากเชื้อรา Alternaria หรือ Ramularia ให้หมั่นดูแลแปลงให้สะอาด ระบายอากาศให้ดี และใช้สารชีวภัณฑ์ไตรโคเดอร์มา หรือสารเคมีกลุ่ม azoxystrobin หากระบาดหนัก"
                icon = "⚪"

            elif result_class == 'Not leaf chilli':
                treatment_text = "ระบบตรวจจับว่าภาพนี้ **ไม่ใช่ใบพริก** หรือภาพไม่ชัดเจน กรุณาถ่ายภาพใบพริกใหม่อีกครั้ง เพื่อการวิเคราะห์ที่แม่นยำ"
                bg_color = "rgba(255, 235, 238, 0.8)" # สีแดงอ่อน
                icon_bg = "#ffcdd2"
                icon = "❌"
            
            # แสดงกล่องคำแนะนำ (HTML Box)
            st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 25px; border-radius: 15px; margin-top: 25px; text-align: left; border: 1px solid rgba(0,0,0,0.05);">
                    <div style="display: flex; align-items: start;">
                        <div style="background: {icon_bg}; width: 45px; height: 45px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.4rem; margin-right: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); flex-shrink: 0;">
                            {icon}
                        </div>
                        <div>
                            <strong style="display: block; margin-bottom: 5px; color: #333; font-size: 1rem;">คำแนะนำการดูแลรักษา</strong>
                            <span style="color: #555; line-height: 1.5; font-size: 0.9rem;">{treatment_text}</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# 6. Footer (เครดิตผู้จัดทำ)
st.markdown("""
    <div class="footer-credit">
        <strong>วิจัยทางคอมพิวเตอร์  โดยสาขาวิชาคอมพิวเตอร์ศึกษา</strong> <br>
        <strong>คณะครุศาสตร์  มหาวิทยาลัยราชภัฏอุบลราชธานี</strong> <br>
        <span class="badge-custom">V.1.0 (Final Release)</span> <br>
        <div style="margin-top: 10px; font-size: 0.75rem; color: #aaa;">
            <strong>พัฒนาโดย: แมวใส่ชุดกบ และผองเพื่อน</strong>
        </div>
    </div>
""", unsafe_allow_html=True)
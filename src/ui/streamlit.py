import streamlit as st
import requests
from PIL import Image
import os

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Skin Disease Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = os.getenv("API_URL", "https://e2e-vision-pipeline-onnx-backend.onrender.com/predict")

# --- 2. CSS TÙY CHỈNH (QUAN TRỌNG ĐỂ TĂNG CỠ CHỮ & MÀU SẮC) ---
st.markdown("""
<style>
    /* Tùy chỉnh Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 2px solid #e9ecef;
    }
    
    /* Box Cảnh báo trong Sidebar - Làm cho nó RỰC RỠ */
    .warning-box {
        background-color: #ffcdd2; /* Nền đỏ nhạt */
        border-left: 10px solid #d32f2f; /* Viền trái đỏ đậm */
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .warning-title {
        color: #b71c1c;
        font-weight: 900;
        font-size: 20px; /* Chữ to */
        margin-bottom: 10px;
        text-transform: uppercase;
    }
    .warning-text {
        color: #333;
        font-size: 16px; /* Chữ nội dung to dễ đọc */
        font-weight: 600;
        line-height: 1.5;
    }

    /* Box Hướng dẫn */
    .guide-box {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 8px;
        border-left: 10px solid #1976d2;
    }
    .guide-text {
        font-size: 16px;
        color: #0d47a1;
        line-height: 1.6;
    }

    /* Tiêu đề Dự án (Header) */
    .main-title {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 3.5rem; /* Rất to */
        font-weight: 800;
        color: #1a237e; /* Màu xanh đậm chuyên nghiệp */
        margin-bottom: 0px;
        line-height: 1.1;
    }
    
    /* Dòng chữ tác giả */
    .author-line {
        font-size: 1.5rem; /* To rõ */
        color: #546e7a;
        margin-bottom: 30px;
        margin-top: 10px;
        border-bottom: 2px solid #eee;
        padding-bottom: 20px;
    }
    .author-name {
        color: #0288d1; /* Màu xanh nổi bật cho tên */
        font-weight: bold;
        text-decoration: underline;
    }

</style>
""", unsafe_allow_html=True)

# --- 3. SIDEBAR (BÊN TRÁI) ---
with st.sidebar:
    # CẢNH BÁO QUAN TRỌNG (Dùng HTML để style mạnh tay)
    st.markdown("""
        <div class="warning-box">
            <div class="warning-title">⚠️ CẢNH BÁO Y TẾ</div>
            <div class="warning-text">
                Kết quả từ AI mang tính chất tham khảo.
                <br><br>
                <b>TUYỆT ĐỐI KHÔNG</b> thay thế chẩn đoán của bác sĩ chuyên khoa.
                <br><br>
                Hãy đến bệnh viện nếu có dấu hiệu bất thường.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # HƯỚNG DẪN SỬ DỤNG
    st.markdown("""
        <div class="guide-box">
            <h3 style="margin-top:0; color:#1565c0;">📖 Hướng dẫn nhanh</h3>
            <ol class="guide-text" style="padding-left: 20px;">
                <li><b>Bước 1:</b> Chụp ảnh vùng da rõ nét.</li>
                <li><b>Bước 2:</b> Tải ảnh lên khung bên phải.</li>
                <li><b>Bước 3:</b> Nhấn nút <b>"Phân tích ngay"</b>.</li>
                <li><b>Bước 4:</b> Đọc kết quả và khuyến cáo.</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

# --- 4. GIAO DIỆN CHÍNH (HEADER & CONTENT) ---

# Header Custom
st.markdown('<div class="main-title">SKIN DISEASE CLASSIFIER</div>', unsafe_allow_html=True)
st.markdown('<div class="author-line">Personal Project Developed by <span class="author-name">PHAT NGUYEN</span></div>', unsafe_allow_html=True)

# Layout 2 cột
col_left, col_right = st.columns([1, 1.2], gap="large")

# --- CỘT 1: UPLOAD ---
with col_left:
    st.subheader("📸 1. Tải hình ảnh")
    st.info("Hỗ trợ định dạng: JPG, PNG, JPEG")
    
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if uploaded_file:
        # Hiển thị ảnh với style bo góc
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown('<style>img {border-radius: 10px;}</style>', unsafe_allow_html=True)
        st.image(image, caption="Ảnh bạn đã chọn", use_column_width=True)

# --- CỘT 2: KẾT QUẢ ---
with col_right:
    st.subheader("🔍 2. Kết quả phân tích")
    
    if uploaded_file:
        # Nút bấm to và nổi bật
        if st.button("🚀 PHÂN TÍCH NGAY", type="primary", use_container_width=True):
            
            with st.spinner("AI đang xử lý hình ảnh... vui lòng chờ..."):
                try:
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    resp = requests.post(API_URL, files=files, timeout=15)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        disease = data.get("disease", "Chưa xác định")
                        confidence = float(data.get("confidence", 0.0))
                        
                        # --- BOX KẾT QUẢ ---
                        st.markdown(f"""
                        <div style="background-color: #f1f8e9; padding: 25px; border-radius: 15px; border: 1px solid #c5e1a5; margin-top: 10px;">
                            <h4 style="color: #33691e; margin:0;">DỰ ĐOÁN CỦA AI:</h4>
                            <h1 style="color: #2e7d32; font-size: 40px; margin: 10px 0;">{disease}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Thanh Confidence
                        st.markdown(f"**Độ tin cậy (Confidence Score):** {confidence:.1%}")
                        st.progress(confidence)
                        
                        # Khuyến cáo dựa trên ngưỡng
                        st.markdown("### 💡 Khuyến cáo hành động:")
                        if confidence > 0.8:
                            st.error("🔴 **NGUY CƠ CAO:** Kết quả có độ tin cậy lớn. Bạn nên đặt lịch khám với bác sĩ da liễu ngay để kiểm tra kỹ lưỡng.")
                        elif confidence > 0.5:
                            st.warning("🟠 **NGHI NGỜ:** AI phát hiện các dấu hiệu tương đồng. Cần theo dõi thêm và tham vấn bác sĩ.")
                        else:
                            st.info("🟢 **CHƯA RÕ RÀNG:** Hình ảnh không đủ cơ sở hoặc không có bệnh lý nguy hiểm. Hãy thử chụp lại rõ nét hơn.")
                            
                    else:
                        st.error(f"Lỗi Server: {resp.status_code}")
                        
                except Exception as e:
                    st.error("Không thể kết nối đến máy chủ AI. Vui lòng kiểm tra lại API.")
    else:
        # Placeholder khi chưa có ảnh
        st.markdown("""
        <div style="text-align: center; padding: 50px; background-color: #f5f5f5; border-radius: 10px; color: #757575;">
            👈 Vui lòng tải ảnh lên ở cột bên trái để bắt đầu
        </div>
        """, unsafe_allow_html=True)
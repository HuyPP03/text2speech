import streamlit as st
import torch
from pathlib import Path
import os

# Đặt tiêu đề cho ứng dụng - phải đặt đầu tiên
st.set_page_config(page_title="Ứng dụng dịch máy Việt-Anh", layout="wide")

# Import thư viện cho các model sẵn có
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sys
import subprocess

# Import mô hình tự tạo nếu có
try:
    from pytorch_lstm_model import load_lstm_model, translate_lstm
    from pytorch_transformer_model import load_transformer_model, translate_transformer
    custom_models_available = True
except ImportError:
    custom_models_available = False

# Tiêu đề cho ứng dụng
st.title("Ứng dụng dịch máy Việt-Anh")

# Kiểm tra thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.sidebar.write(f"Đang sử dụng thiết bị: **{device}**")

# Các đường dẫn model tự tạo
lstm_path = 'lstm_model.pth'
transformer_path = 'transformer_model.pth'

has_lstm = Path(lstm_path).exists() and custom_models_available
has_transformer = Path(transformer_path).exists() and custom_models_available

# Danh sách các model có sẵn
pretrained_models = [
    "Helsinki-NLP/opus-mt-vi-en",
]

# Lưu các model để sử dụng
helsinki_model = None
helsinki_tokenizer = None

# Tạo danh sách các model có thể sử dụng
available_models = []

if has_lstm:
    available_models.append("LSTM với Attention (Custom)")
    
if has_transformer:
    available_models.append("Transformer (Custom)")

available_models.append("Helsinki-NLP/opus-mt-vi-en (Pre-trained)")

# Tải model Helsinki-NLP
@st.cache_resource
def load_helsinki_model():
    with st.spinner("Đang tải model Helsinki-NLP..."):
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-vi-en")
        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-vi-en").to(device)
        return model, tokenizer

# Tải các model tự tạo
@st.cache_resource
def load_custom_models():
    models = {}
    if has_lstm:
        with st.spinner("Đang tải model LSTM..."):
            lstm_model, vi_vocab_lstm, en_vocab_lstm, idx_to_vi_lstm, idx_to_en_lstm = load_lstm_model(lstm_path, device)
            models["LSTM với Attention (Custom)"] = {
                "model": lstm_model,
                "vi_vocab": vi_vocab_lstm,
                "en_vocab": en_vocab_lstm,
                "idx_to_vi": idx_to_vi_lstm,
                "idx_to_en": idx_to_en_lstm
            }
    
    if has_transformer:
        with st.spinner("Đang tải model Transformer..."):
            transformer_model, vi_vocab_trans, en_vocab_trans, idx_to_vi_trans, idx_to_en_trans = load_transformer_model(transformer_path, device)
            models["Transformer (Custom)"] = {
                "model": transformer_model,
                "vi_vocab": vi_vocab_trans,
                "en_vocab": en_vocab_trans,
                "idx_to_vi": idx_to_vi_trans,
                "idx_to_en": idx_to_en_trans
            }
    
    return models

# Chọn mô hình để sử dụng
if available_models:
    # Chọn mô hình
    selected_model = st.sidebar.radio("Chọn mô hình dịch thuật:", available_models)
    
    # Load các model tùy thuộc vào lựa chọn
    custom_models = None
    if has_lstm or has_transformer:
        custom_models = load_custom_models()
    
    if "Helsinki-NLP" in selected_model:
        helsinki_model, helsinki_tokenizer = load_helsinki_model()
    
    # Giao diện người dùng để nhập văn bản
    st.header("Nhập văn bản tiếng Việt để dịch")
    
    # Sử dụng session_state để lưu trữ giá trị input_text
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""
        
    input_text = st.text_area("Văn bản tiếng Việt:", height=150, value=st.session_state.input_text)
    
    # Nút dịch
    if st.button("Dịch"):
        if input_text:
            with st.spinner(f"Đang dịch với mô hình {selected_model}..."):
                translation = ""
                
                # Thực hiện dịch thuật với mô hình đã chọn
                if "LSTM với Attention" in selected_model:
                    model_data = custom_models[selected_model]
                    translation = translate_lstm(model_data["model"], input_text, model_data["vi_vocab"], model_data["en_vocab"], model_data["idx_to_en"], device)
                
                elif "Transformer (Custom)" in selected_model:
                    model_data = custom_models[selected_model]
                    translation = translate_transformer(model_data["model"], input_text, model_data["vi_vocab"], model_data["en_vocab"], model_data["idx_to_en"], device)
                
                elif "Helsinki-NLP" in selected_model:
                    inputs = helsinki_tokenizer(input_text, return_tensors="pt").to(device)
                    output_ids = helsinki_model.generate(**inputs)
                    translation = helsinki_tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                # Hiển thị kết quả
                st.header("Kết quả dịch (tiếng Anh)")
                st.write(translation)
                
                # Hiển thị một số thống kê
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Độ dài văn bản gốc:** {len(input_text.split())} từ")
                with col2:
                    st.write(f"**Độ dài văn bản dịch:** {len(translation.split())} từ")
        else:
            st.warning("Vui lòng nhập văn bản để dịch.")
    
    # Thông tin về các mô hình
    st.sidebar.header("Thông tin mô hình")
    
    if "LSTM với Attention" in selected_model:
        st.sidebar.write("""
        **Mô hình LSTM với Attention (Custom)**
        - Sử dụng mạng LSTM hai chiều để mã hóa câu nguồn
        - Sử dụng cơ chế Attention để tập trung vào các phần quan trọng của câu nguồn
        - Phù hợp với các câu ngắn và trung bình
        """)
    elif "Transformer (Custom)" in selected_model:
        st.sidebar.write("""
        **Mô hình Transformer (Custom)**
        - Dựa trên kiến trúc Transformer gốc từ bài báo "Attention Is All You Need"
        - Sử dụng cơ chế Self-Attention để nắm bắt ngữ cảnh toàn cầu
        - Phù hợp với cả câu ngắn và dài
        - Thường cho kết quả dịch tốt hơn với các cấu trúc phức tạp
        """)
    elif "Helsinki-NLP" in selected_model:
        st.sidebar.write("""
        **Mô hình Helsinki-NLP/opus-mt-vi-en**
        - Mô hình được huấn luyện trên bộ dữ liệu OPUS
        - Sử dụng kiến trúc Transformer nhỏ gọn
        - Mô hình được tối ưu hóa cho hiệu suất và tốc độ
        - Nguồn: https://huggingface.co/Helsinki-NLP/opus-mt-vi-en
        """)
    
    # Thêm các ví dụ để người dùng có thể thử
    st.sidebar.header("Các ví dụ")
    
    examples = [
        "Xin chào, tôi đang học cách xây dựng mô hình dịch máy.",
        "Học máy là một lĩnh vực của trí tuệ nhân tạo.",
        "Phương pháp này sử dụng kiến trúc Transformer để nâng cao chất lượng dịch thuật."
    ]
    
    for i, example in enumerate(examples):
        if st.sidebar.button(f"Ví dụ {i+1}"):
            st.session_state.input_text = example
            st.experimental_rerun()
else:
    st.warning("Không có mô hình nào được tải. Đang tải các mô hình có sẵn...")
    with st.spinner("Đang tải mô hình Helsinki-NLP..."):
        helsinki_model, helsinki_tokenizer = load_helsinki_model()
    st.success("Đã tải mô hình Helsinki-NLP thành công! Vui lòng làm mới trang để sử dụng.")

# Thêm thông tin về tác giả
st.sidebar.markdown("---")
st.sidebar.markdown("### Thông tin")
st.sidebar.markdown("**Ứng dụng dịch máy Việt-Anh**")
st.sidebar.markdown("Sử dụng PyTorch, Transformers và Streamlit")
st.sidebar.markdown("Các mô hình: Custom, Helsinki-NLP")
import streamlit as st
import os
import time
import base64
from gtts import gTTS
from datetime import datetime

# Thiết lập tiêu đề ứng dụng
st.set_page_config(page_title="Chuyển đổi Văn bản tiếng Việt sang Giọng nói", layout="wide")
st.title("Chuyển đổi Văn bản tiếng Việt sang Giọng nói")

# Tạo thư mục lưu trữ nếu chưa tồn tại
output_dir = "output_audio"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function để tạo âm thanh từ văn bản
def text_to_speech(text, lang='vi', tld='com.vn', slow=False):
    tts = gTTS(text=text, lang=lang, tld=tld, slow=slow)
    return tts

# Function để mã hóa file audio thành base64 cho HTML
def get_audio_base64(file_path):
    with open(file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
        base64_audio = base64.b64encode(audio_bytes).decode()
        return base64_audio

# Function để tạo HTML audio với điều khiển tốc độ
def get_html_audio_player(file_path, default_speed=1.0):
    base64_audio = get_audio_base64(file_path)
    html = f"""
    <audio id="audio-player" controls>
        <source src="data:audio/mp3;base64,{base64_audio}" type="audio/mp3">
        Trình duyệt của bạn không hỗ trợ phát audio.
    </audio>
    <br>
    <div>
        <label for="speed-control">Tốc độ phát: <span id="speed-value">{default_speed}x</span></label>
        <input type="range" id="speed-control" min="0.5" max="3" step="0.1" value="{default_speed}" style="width:300px;">
    </div>
    <script>
        // Đặt tốc độ mặc định ngay khi tải trang
        const audio = document.getElementById('audio-player');
        const speedControl = document.getElementById('speed-control');
        const speedValue = document.getElementById('speed-value');
        
        // Thiết lập tốc độ mặc định ngay lập tức
        audio.playbackRate = {default_speed};
        
        speedControl.addEventListener('input', function() {{
            const speed = parseFloat(this.value);
            audio.playbackRate = speed;
            speedValue.textContent = speed + 'x';
        }});
    </script>
    """
    return html

# Sidebar để điều khiển chức năng ứng dụng
with st.sidebar:
    st.header("Cài đặt")
    
    # Lựa chọn giọng đọc
    voice_options = {
        "Tiếng Việt (Nữ miền Nam)": "com.vn",
        "Tiếng Việt (Nữ miền Bắc)": "com",
        "Tiếng Việt (Google US)": "us",
        "Tiếng Việt (Google UK)": "co.uk",
        "Tiếng Việt (Google Ấn Độ)": "co.in",
        "Tiếng Việt (Google Úc)": "com.au",
    }
    selected_voice = st.selectbox("Chọn giọng đọc", list(voice_options.keys()))
    voice_tld = voice_options[selected_voice]
    
    # Tùy chọn tốc độ đọc gốc (gTTS)
    use_slow_mode = st.checkbox("Sử dụng chế độ chậm của gTTS", value=False,
                             help="Sử dụng chế độ chậm có sẵn của gTTS")
    
    # Tốc độ phát cho trình phát HTML
    default_playback_speed = st.slider(
        "Tốc độ phát mặc định",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Điều chỉnh tốc độ phát mặc định (0.5x đến 3.0x)"
    )

# Giao diện chính
st.header("Nhập văn bản tiếng Việt")
text_input = st.text_area("", height=150, 
                        placeholder="Nhập văn bản tiếng Việt ở đây...", 
                        help="Nhập văn bản tiếng Việt bạn muốn chuyển thành giọng nói")

# Đặt tên cho file âm thanh
audio_filename = st.text_input("Tên file (không cần phần mở rộng)", 
                            placeholder="audio_output",
                            help="Nhập tên cho file âm thanh xuất ra")

if not audio_filename:
    audio_filename = "audio_output"

# Nút tạo âm thanh
if st.button("Tạo giọng nói"):
    if text_input:
        with st.spinner("Đang xử lý..."):
            try:
                # Tạo âm thanh
                speech_output = text_to_speech(text_input, lang='vi', tld=voice_tld, slow=use_slow_mode)
                
                # Tạo tên file với timestamp
                timestamp = int(time.time())
                filename = f"{audio_filename}_{timestamp}.mp3"
                
                # Lưu file âm thanh
                filepath = os.path.join(output_dir, filename)
                speech_output.save(filepath)
                
                # Hiển thị trình phát HTML với điều khiển tốc độ
                html_player = get_html_audio_player(filepath, default_playback_speed)
                st.components.v1.html(html_player, height=120)
                
                # Lưu thông tin về file vừa tạo
                if 'created_files' not in st.session_state:
                    st.session_state.created_files = {}
                
                st.session_state.created_files[filename] = filepath
                
                # Nút tải xuống và xóa file
                with open(filepath, "rb") as file:
                    audio_bytes = file.read()
                
                if st.download_button(
                    label="Tải xuống và xóa file",
                    data=audio_bytes,
                    file_name=filename,
                    mime="audio/mp3",
                    key=f"download_{timestamp}"
                ):
                    # Logic xóa file sẽ được xử lý sau khi tải xuống
                    # Do streamlit không hỗ trợ trực tiếp callback after download
                    # Chúng ta sẽ thêm một nút xác nhận riêng để xóa
                    st.session_state.file_to_delete = filepath
                
                st.success(f"Đã tạo file âm thanh: {filename}")
                
                # Nút xác nhận xóa sau khi tải xuống
                if st.button("Xác nhận đã tải xuống và xóa file"):
                    if 'file_to_delete' in st.session_state and os.path.exists(st.session_state.file_to_delete):
                        try:
                            os.remove(st.session_state.file_to_delete)
                            st.success(f"Đã xóa file sau khi tải xuống")
                            # Xóa thông tin file khỏi session_state
                            for key, value in list(st.session_state.created_files.items()):
                                if value == st.session_state.file_to_delete:
                                    del st.session_state.created_files[key]
                            del st.session_state.file_to_delete
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Lỗi khi xóa file: {e}")
                
            except Exception as e:
                st.error(f"Lỗi khi tạo giọng nói: {e}")
    else:
        st.warning("Vui lòng nhập văn bản để tạo giọng nói")

# Hiển thị danh sách các file âm thanh đã tạo
st.header("Các file âm thanh đã tạo")
audio_files = [f for f in os.listdir(output_dir) if f.endswith('.mp3')]

if audio_files:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_file = st.selectbox("Chọn file để nghe", audio_files)
        selected_path = os.path.join(output_dir, selected_file)
        
        # Hiển thị trình phát với điều khiển tốc độ cho file đã chọn
        html_player = get_html_audio_player(selected_path, default_playback_speed)
        st.components.v1.html(html_player, height=120)
        
        # Nút tải xuống file đã chọn
        with open(selected_path, "rb") as file:
            st.download_button(
                label="Tải xuống file đã chọn",
                data=file,
                file_name=selected_file,
                mime="audio/mp3",
                key="download_existing"
            )
    
    with col2:
        # Nút xóa file đã chọn
        if st.button("Xóa file đã chọn"):
            try:
                os.remove(selected_path)
                st.success(f"Đã xóa file: {selected_file}")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Lỗi khi xóa file: {e}")
else:
    st.info("Chưa có file âm thanh nào được tạo")

# Thông tin cuối trang
st.markdown("---")
st.markdown("### Hướng dẫn sử dụng")
st.markdown("""
1. Chọn giọng đọc và tốc độ phát mặc định ở mục cài đặt
2. Nhập văn bản tiếng Việt vào ô trên
3. Đặt tên cho file âm thanh (tùy chọn)
4. Nhấn nút "Tạo giọng nói" để chuyển đổi
5. Điều chỉnh tốc độ phát với thanh trượt bên dưới trình phát
6. Tải xuống và xóa file nếu cần
""")

# Thêm giải thích về các tùy chọn
st.markdown("### Thông tin thêm")
st.markdown("""
- **Giọng đọc**: Mỗi giọng đọc sẽ có âm điệu và cách phát âm khác nhau
- **Chế độ chậm của gTTS**: Tạo giọng nói với tốc độ chậm hơn ban đầu
- **Tốc độ phát mặc định**: Điều chỉnh tốc độ phát từ 0.5x (rất chậm) đến 3.0x (rất nhanh)
- **Thanh trượt tốc độ**: Cho phép điều chỉnh tốc độ phát trực tiếp khi nghe
""")
import streamlit as st
from ocr_reader import extract_text
from PIL import Image
import os
import re

# To create upload directory if not exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Function to highlight keywords in text
def highlight_keywords(text, keyword):
    highlighted_text = re.sub(f'({keyword})', r'<span style="background-color: yellow;">\1</span>', text, flags=re.IGNORECASE)
    return highlighted_text

# Streamlit App
st.title("OCR: Text Extractor (Using ColPali, Byaldi, and Qwen2-VL)")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Initialize session state for extracted text
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""

if uploaded_file is not None:
    
    image_path = os.path.join("uploads", uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display uploaded image
    image = Image.open(image_path)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Perform OCR only if extracted text is empty
    if st.session_state.extracted_text == "":
        st.write("Extracting text using ColPali and Byaldi with Qwen2-VL...")
        st.session_state.extracted_text = extract_text(image_path)

    # Display extracted text
    st.text_area("Extracted Text", st.session_state.extracted_text, height=200)
    
    # Search functionality: Keyword input
    keyword = st.text_input("Enter keyword to highlight")
    
    if keyword:
        
        highlighted_text = highlight_keywords(st.session_state.extracted_text, keyword)
        
        st.markdown(f"<div style='white-space: pre-wrap;'>{highlighted_text}</div>", unsafe_allow_html=True)


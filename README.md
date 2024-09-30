# OCR and Document Search Web Application

 Website: https://ocr-reader-colpali.streamlit.app/

## Overview
This project is a web-based prototype that demonstrates the ability to perform Optical Character Recognition (OCR) on uploaded images. The application allows users to upload an image, extract text, and perform a keyword search within the extracted text.

## Features
- Upload an image file (JPEG, PNG).
- Extract text using OCR models (ColPali, Byaldi, Qwen2-VL).
- Highlight keywords in the extracted text.
- User-friendly interface built with Streamlit.

## Technologies Used
- Python
- Streamlit
- Huggingface Transformers
- PyTorch
- ColPali, Byaldi, Qwen2-VL

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone <your-github-repo-url>
   cd <your-repo-directory>

2. **Create and Activate a Virtual Environment (optional)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   
3. **Install Required Libraries**:
   Make sure you have requirements.txt in your project directory. If not, create it with the necessary packages.

   ```bash
   pip install -r requirements.txt
   
4. **Run the Application**:
   ```bash
   streamlit run app.py
   
Access the Application: Open your web browser and go to http://localhost:8501.


 Screenshots:
 
 ![image](https://github.com/user-attachments/assets/3f8b3c37-c42a-4ea3-b773-8f4e66c7ec21)
 
 ![image](https://github.com/user-attachments/assets/ec85cbe0-41a1-4d0b-a633-f0b7c189825e)
 
 ![image](https://github.com/user-attachments/assets/ea05c996-11e8-41bf-8f1c-7a08d261567d)

 ![image](https://github.com/user-attachments/assets/21792af5-47f4-42d2-9fd7-d622aa7f8a0d)

 Note: It takes 15-20 minutes to get results on website or localhost.



 

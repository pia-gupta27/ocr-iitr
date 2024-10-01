import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from byaldi import RAGMultiModalModel
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


# Load ColPali model and processor
colpali_model_name = "vidore/colpali-v1.2"
colpali_model = ColPali.from_pretrained(colpali_model_name, torch_dtype=torch.bfloat16, device_map="cpu").eval()
colpali_processor = AutoProcessor.from_pretrained(colpali_model_name)


# Load Byaldi model
rag_model = RAGMultiModalModel.from_pretrained(colpali_model_name, index_root=".byaldi/")


# Load Qwen2-VL model and processor
qwen_model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto")
qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")


def extract_text(image_path):
    print("Loading image...")
    image = Image.open(image_path).convert("RGB")

    # Resize the image to reduce memory usage
    image = image.resize((image.width // 4, image.height // 4))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    colpali_model.to(device)
    qwen_model.to(device)

    # Step 1: Process image using ColPali processor
    print("Processing image using ColPali...")
    inputs = colpali_processor(images=[image], return_tensors="pt").to(device)

    # Step 2: Pass inputs through ColPali model
    print("Passing inputs to ColPali model...")
    with torch.no_grad():
        image_embedding = colpali_model(**inputs)
    print("Image embedding generated.")

    # Step 3: Prepare message for Qwen2-VL
    print("Preparing message for Qwen2-VL...")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Extract the text from this image."},
            ],
        }
    ]

    # Step 4: Generate OCR text using Qwen2-VL
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)

    print("Generating OCR text using Qwen2-VL...")
    with torch.no_grad():
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=256)
        output_text = qwen_processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    extracted_text = output_text[0]
    print("OCR text extracted:", extracted_text)

    return extracted_text

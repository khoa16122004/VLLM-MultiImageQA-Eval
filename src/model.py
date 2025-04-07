import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel as HFCLIPModel

class MyCLIPWrapper:
    def __init__(self, model_name='openai/clip-vit-base-patch32', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = HFCLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def visual_encode(self, image_input):
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError("Unsupported input type for image")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features.squeeze().cpu().numpy()

    def text_encode(self, text):
        # Giới hạn số lượng token
        max_length = 77  # Giới hạn chiều dài chuỗi

        # Cắt ngắn văn bản nếu cần
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(self.device)

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        return text_features.squeeze().cpu().numpy()

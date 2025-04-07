import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel as HFCLIPModel
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../ReT")))
from src.models import RetrieverModel, RetModel

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

class ReTWrapper:
    def __init__(self):
        retrieval = RetrieverModel.from_pretrained('aimagelab/ReT-CLIP-ViT-L-14', device_map="cuda") # E_Qs
        self.encode: RetModel = retrieval.get_query_model()
        self.encode.init_tokenizer_and_image_processor()
        
        self.query: RetModel = retrieval.get_passage_model()
        self.query.init_tokenizer_and_image_processor()
    
    def encode(self, img, txt=""): # img: path, txt: str
        if txt:
            ret_feats = self.query.get_ret_features([[txt, img]]).squeeze(0)
        else: # txt = ""
            ret_feats = self.encode.get_ret_features([[txt, img]]).squueze(0)
        
        return ret_feats.cpu().numpy()
    
    def sim(self, q1, q2): # q1: ret_feats, q2: ret_feats
        # q1: 32 x d
        # q2: 32 x d
        # sim = sigma_i -> 32 (max(sigma_j -> 32 (q1_i * q2_j)))
        
        return (torch.matmul(q1, q2.T)).max(dim=1).values.sum()

            
        
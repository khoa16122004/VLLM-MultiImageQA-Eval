import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

class CLIPModel:
    def __init__(self, model_name='openai/clip-vit-base-patch32', device=None):
        '''
        Args:
            model_name: name of CLIP model from HuggingFace
            device: cuda or cpu (optional)
        '''
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def visual_encode(self, image_input):
        '''
        Args:
            image_input: image path (str) or PIL.Image
        Returns:
            torch.Tensor: visual embedding
        '''
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
        '''
        Args:
            text: str or list of str
        Returns:
            torch.Tensor: text embedding(s)
        '''
        inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features.squeeze().cpu().numpy()

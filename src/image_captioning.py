from models.llava_ import LLava
from PIL import Image
import os
from tqdm import tqdm

image_token = "<image>"
output_dir = "caption_corpus"
os.makedirs(output_dir, exist_ok=True)  

model = LLava("llava-onevision-qwen2-7b-ov", "llava_qwen")

dataset_path = "../dataset/MRAG_corpus"
caption_prompt = "Describe the image in great detail, mentioning every visible element, their appearance, location, and how they interact in the scene. <image>"

for file_name in tqdm(os.listdir(dataset_path)):
    file_path = os.path.join(dataset_path, file_name)
    img = [Image.open(file_path).convert("RGB")]
    output = model.inference(caption_prompt, img)[0]
    
    with open(os.path.join(output_dir, file_name + ".txt"), "w") as f:
        f.write(output)
from extraction import CreateDatabase
from model import MyCLIPWrapper
from PIL import Image
import os
from models.llava_ import LLava
def extract_question(sample_dir):
    gt_files = []
    for img_name in os.listdir(sample_dir):
        if "question" in img_name and img_name.endswith(".png"):
            question_img = Image.open(os.path.join(sample_dir, img_name)).convert('RGB')
            
        elif "gt" in img_name:
            gt_files.append(os.path.join(sample_dir, img_name))
        
        else:
            with open(os.path.join(sample_dir, img_name), "r") as f:
                question = f.read().strip()
                choices = []
                
                f.read()
                for i in range(4):
                    choices.append(f.read().strip())
                gt_ans = f.read().strip().split("Anwers: ")
    
    return question, question_img, gt_files, choices, gt_ans
                
            

def main():
    model = MyCLIPWrapper()
    db = CreateDatabase(model=model)
    
    dataset_dir = "../dataset/MRAG"
    database_dir = "../database/MRAG"
    
    lvlm = LLava(args.pretrained, args.model_name)
    prefix_question = "You will be given one question concerning several images. The first image is the input image, others are retrieved examples to help you. Answer with the option's letter from the given choices directly."
    image_token = "<image>"
    # retrieval
    for sample_id in os.listdir(dataset_dir):
        if sample_id != "index" and not sample_id.endswith(".py"):
            sample_dir = os.path.join(dataset_dir, sample_id)
            question, question_img, gt_files, choices, gt_ans = extract_question(sample_dir)
            num_input_images = len(gt_files) + 1
            choice_join = choices.join("\n")
            full_question = f"{prefix_question}{num_input_images * image_token}\n{question}\n{choice_join}"
            print(full_question)
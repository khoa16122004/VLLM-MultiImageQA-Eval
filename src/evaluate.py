from extraction import CreateDatabase
from model import MyCLIPWrapper
from PIL import Image
import os
import argparse
import numpy as np
from models.llava_ import LLava
from tqdm import tqdm
import random
import torch
from gpt_extract import extract_answer

def extract_question(sample_dir):
    gt_files = []
    for img_name in os.listdir(sample_dir):
        if "question" in img_name and img_name.endswith(".png"):
            question_img = Image.open(os.path.join(sample_dir, img_name)).convert('RGB')
            
        elif "gt" in img_name:
            gt_files.append(os.path.join(sample_dir, img_name))
        
        else:
            with open(os.path.join(sample_dir, img_name), "r") as f:
                question = f.readline().strip()
                choices = []
                
                f.readline()
                for i in range(4):
                    choices.append(f.readline().strip())
                
                f.readline()
                gt_ans = f.readline().strip().split("Anwers: ")[1]
    return question, question_img, gt_files, choices, gt_ans

def extract_output(out, prompt):
    if out not in ['A', 'B', 'C', 'D']:
        extraction  = extract_answer(out, prompt)
        out = extraction

    return out

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True   
            
def init_model(args):
    special_token = None
    if "llava" in args.model_name:
        from models.llava_ import LLava
        image_token = "<image>"
        model = LLava(args.pretrained, args.model_name)

    elif "openflamingo" in args.model_name:
        from models.openflamingo_ import OpenFlamingo
        image_token = "<image>"
        special_token = "<|endofchunk|>"

        model = OpenFlamingo(args.pretrained)
    
    elif "mantis" in args.model_name:
        from models.mantis_ import Mantis
        image_token = "<image>"
        model = Mantis(args.pretrained)
        
    elif "deepseek" in args.model_name:
        from models.deepseek_ import DeepSeek
        image_token = "<image_placeholder>"
        model = DeepSeek(args.pretrained)
    
    return model, image_token, special_token

def main(args):
    seed_everything(22520691)
    model = MyCLIPWrapper()
    db = CreateDatabase(model=model)
    
    
    dataset_dir = "../dataset/MRAG"
    database_dir = "../database/MRAG_CLIP/index"
    
    lvlm, image_token, special_token = init_model(args)
    retrieved_prefix_question = "You will be given one question concerning several images. The first image is the input image, others are retrieved examples to help you. Answer with the option's letter from the given choices directly."
    no_retrieved_prefix_question = "You will be given one question concerning one image. Answer with the option's letter from the given choices directly."
    # retrieval
    is_contain_retrieval = 0
    retrieved_acc = 0
    acc = 0
    num_samples = 0
    for sample_id in tqdm(os.listdir(dataset_dir)):
        if sample_id != "index" and not sample_id.endswith(".py"):
            num_samples += 1
            sample_dir = os.path.join(dataset_dir, sample_id)
            question, question_img, gt_files, choices, gt_ans = extract_question(sample_dir)
            
            # retrieved output
            num_input_images = len(gt_files) + 1
            choice_join = "\n".join(choices)
            full_question = f"{retrieved_prefix_question}{num_input_images * image_token}\n{question}\n{choice_join}"
            retrieved_paths, retrieved_sample_ids = db.combined_search(index_dir=database_dir, dataset_dir=dataset_dir, 
                                                                   image_index=int(sample_id), k=args.topk, 
                                                                   topk_rerank=args.topk_rerank)
            retrieved_files = [Image.open(path).convert("RGB") for path in retrieved_paths]
            output = lvlm.inference(full_question, [question_img, *retrieved_files])[0]
            output = extract_output(output, question)
            if np.any([int(sample_id) == retrieved_sample_id for retrieved_sample_id in retrieved_sample_ids]):
                is_contain_retrieval = 1
            if gt_ans == output:
                retrieved_acc += 1    
                
                
            # without retrieval
            num_input_images = 1
            full_question = f"{no_retrieved_prefix_question}{num_input_images * image_token}\n{question}\n{choice_join}"
            output = lvlm.inference(full_question, [question_img])[0]
            output = extract_output(output, question)
            if gt_ans == output:
                acc += 1
                
    print(f"Accuracy without retrieval: {acc / num_samples * 100}%, Accuracy with retrieval: {retrieved_acc / num_samples * 100}%, Contain retrieval: {is_contain_retrieval}%, Total samples: {num_samples}")            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, default="llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    parser.add_argument("--topk_rerank", type=int, default=50)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()
    
    main(args)
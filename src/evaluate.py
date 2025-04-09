from extraction import CreateDatabase
from model import MyCLIPWrapper, ReTWrapper
from PIL import Image
import os
import argparse
import numpy as np
from tqdm import tqdm
import random
import torch
from gpt_extract import extract_answer
import json

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
    model_encode = ReTWrapper()
    db = CreateDatabase(model=model_encode, model_name="ReT")
    
    question_dir = "../dataset/MRAG"
    dataset_dir = "../dataset/MRAG_corpus"
    database_dir = "../database/MRAG_corpus_ReT"
    index_dir = "../database/MRAG_corpus_ReT/index"
    
    if args.retrieved_path is not None:
        with open(args.retrieved_path, "r") as f:
            retrieved_data = json.load(f)
    
    
    lvlm, image_token, special_token = init_model(args)
    retrieved_prefix_question = "You will be given one question concerning several images. The first image is the input image, others are retrieved examples to help you. Answer with the option's letter from the given choices directly."
    no_retrieved_prefix_question = "You will be given one question concerning one image. Answer with the option's letter from the given choices directly."
    # retrieval
    # is_contain_retrieval = 0
    acc = 0
    num_samples = 0
    for sample_id in tqdm(os.listdir(question_dir)):
        if sample_id != "index" and not sample_id.endswith(".py"):
            if args.sample_id_eval >= 0:
                if int(sample_id) != args.sample_id_eval:
                    continue
            
            num_samples += 1
            sample_dir = os.path.join(question_dir, sample_id)
            question, question_img, gt_files, choices, gt_ans = extract_question(sample_dir)
            # retrieved output
            choice_join = "\n".join(choices)
            if args.retrieved_path is not None:
                print("Using path results")
                retrieved_paths = retrieved_data[str(sample_id)]
                print(retrieved_paths)
                    
                print("Retrieval paths: ", retrieved_paths)
                retrieved_files = [Image.open(os.path.join(dataset_dir, path)).convert("RGB") for path in retrieved_paths]
                num_input_images = len(retrieved_files) + 1
                full_question = f"{retrieved_prefix_question}{num_input_images * image_token}\n{question}\n{choice_join}"
                print("Question: ", full_question)
                output = lvlm.inference(full_question, [question_img, *retrieved_files])[0]
                # print("Output: ", output)
                # output = extract_output(output, question)
                print("Output: ", output)
                print("Ground truth:", gt_ans)

                if gt_ans == output:
                    acc += 1    

            else:   
                print("Not using retrieval")
                num_input_images = 1
                full_question = f"{no_retrieved_prefix_question}{num_input_images * image_token}\n{question}\n{choice_join}"
                output = lvlm.inference(full_question, [question_img])[0]
                print("Output: ", output)
                output = extract_output(output, question)
                print("Ground truth:", gt_ans)

                if gt_ans == output:
                    acc += 1    




    with open(f"result_{args.model_name}_{args.pretrained}_{args.topk}_{args.topk_rerank}.txt", "a") as f:          
        if args.using_retrieval == True:
            line = f"Accuracy with retrieval: {acc / num_samples * 100}%, Total samples: {num_samples}\n"
            print(line)
        else:
            line = f"Accuracy without retrieval: {acc / num_samples * 100}%, Total samples: {num_samples}\n"
            print(line)
        f.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, default="llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    parser.add_argument("--topk_rerank", type=int, default=10)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--sample_id_eval", type=int, default=-1)
    parser.add_argument("--retrieved_path", type=str, default=None)
    
    args = parser.parse_args()
    
    main(args)
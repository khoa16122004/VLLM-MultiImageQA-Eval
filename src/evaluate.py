from PIL import Image
import os
import argparse
import numpy as np
from tqdm import tqdm
import random
import torch
from gpt_extract import extract_answer
import json
from utils import init_dataset, init_lvlm_model, seed_everything, extract_output

def main(args):
    seed_everything(22520691)

    # load retrieved path
    if args.retrieved_path is not None:
        with open(args.retrieved_path, "r") as f:
            retrieved_data = json.load(f)
    
    # init lvlm model and dataset
    lvlm, image_token, special_token = init_lvlm_model(args.pretrained, args.model_name)
    dataset = init_dataset(args.dataset_name)

    # prefix guide for question
    retrieved_prefix_question = "You will be given one question concerning several images. The first image is the input image, others are retrieved examples to help you. Answer with the option's letter from the given choices directly."
    no_retrieved_prefix_question = "You will be given one question concerning one image. Answer with the option's letter from the given choices directly."

    # evaluate vllm
    acc = 0
    num_samples = 0
    results = {
        'accuracy': None,
        'log_answer': [],
    }
    for (id, question, question_img, gt_files, choices, gt_ans) in tqdm(dataset.loader()):
            num_samples += 1
            
            choice_join = "\n".join(choices)
            if args.retrieved_path is not None:
                retrieved_paths = retrieved_data[id]
                retrieved_files = [Image.open(os.path.join(args.dataset_dir, path)).convert("RGB") for path in retrieved_paths]
                num_input_images = len(retrieved_files) + 1
                full_question = f"{retrieved_prefix_question}{num_input_images * image_token}\n{question}\n{choice_join}"
                output = lvlm.inference(full_question, [question_img, *retrieved_files])[0]
                print("Output: ", output)


 

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
            results["log_answer"][id] = {
                'output': output,
                'gt': gt_ans
            }


    results['accuracy'] = acc * 100 / num_samples
    with open(args.log_output_path, "w") as f:
        json.dump(results, f, indent=4)        
    
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, default="llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    parser.add_argument("--retrieved_path", type=str, default=None)
    parser.add_argument("--log_output_path", type=str)
    parser.add_argument("--dataset_name", type=str, default="MRAG")    
    args = parser.parse_args()
    
    main(args)
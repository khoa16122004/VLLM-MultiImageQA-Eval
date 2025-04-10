import os
import numpy as np
from model import MyCLIPWrapper, ReTWrapper
from tqdm import tqdm
import faiss
import csv
import pandas as pd
from PIL import Image
import argparse
from extraction import CreateDatabase
import json
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
                question = f.readline().strip()
                choices = []
                
                f.readline()
                for i in range(4):
                    choices.append(f.readline().strip())
                
                f.readline()
                gt_ans = f.readline().strip().split("Anwers: ")[1]
    return question, question_img, gt_files, choices, gt_ans


def main(args):
    # encode model
    if args.model_name_encode == "ReT":
        model_encode = ReTWrapper()
    elif args.model_name_encode == "CLIP":
        model_encode = MyCLIPWrapper()
    
    # fliter model
  
    # db
    db = CreateDatabase(index_dir=args.index_dir,
                        dataset_dir=args.dataset_dir,
                        model=model_encode,
                        model_name=args.model_name_encode,
                        model_filter=None,
                        caption_model=None)
    
    results = {}
    # extract retrieval paths
            
    sample_dir = os.path.join(args.question_dir, args.sample_id_eval)
    question, question_img, gt_files, choices, gt_ans = extract_question(sample_dir)
    # retrieved output
    choice_join = "\n".join(choices)
    full_question = f"{question}\n{choice_join}"
    img = Image.open(args.input_path).convert("RGB")
    retrieval_paths = db.flow_search(img, args.question_dir, filter=0, k=args.topk, topk_rerank=args.topk_rerank)
    
    print("Id: ", args.sample_id_eval)            
    print("question: ", question)
    print("retrieval paths: ", retrieval_paths)
    print("--------------------")
    # write to json    
    results[str(args.sample_id_eval)] = retrieval_paths
    output_path = f"test.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_encode", type=str, default="ReT")
    parser.add_argument("--topk_rerank", type=int, default=10)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--sample_id_eval", type=int, default=-1)
    parser.add_argument("--using_retrieval", type=int, default=1)
    parser.add_argument("--question_dir", type=str, default="../dataset/MRAG")
    parser.add_argument("--dataset_dir", type=str, default="../dataset/MRAG_corpus")
    parser.add_argument("--index_dir", type=str, default="../database/MRAG_corpus_ReT_caption/index")
    parser.add_argument("--input_path", type=str)
    args = parser.parse_args()
    main(args)
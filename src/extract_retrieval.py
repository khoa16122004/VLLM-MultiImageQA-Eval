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
    
    # db
    db = CreateDatabase(model=model_encode, 
                        model_name=args.model_name_encode)
    
    results = {}
    # extract retrieval paths
    for sample_id in tqdm(os.listdir(args.question_dir)):
        if sample_id != "index" and not sample_id.endswith(".py"):
            if args.sample_id_eval >= 0:
                if int(sample_id) != args.sample_id_eval:
                    continue
            
            sample_dir = os.path.join(args.question_dir, sample_id)
            question, question_img, gt_files, choices, gt_ans = extract_question(sample_dir)
            # retrieved output
            choice_join = "\n".join(choices)
            retrieval_paths = db.flow_search(index_dir=args.index_dir, 
                                             dataset_dir=args.question_dir,
                                             image_index=int(sample_id),
                                             question=question,
                                             k=args.topk,
                                             topk_rerank=args.topk_rerank)
                                
            print("question: ", question)
            print("retrieval paths: ", retrieval_paths)
            print("--------------------")
            # write to json    
            results[str(sample_id)] = retrieval_paths     

    output_path = f"mrag_bench_top{args.topk}_modelencode={args.model_name_encode}_modelcaption={args.model_name_caption}_topk_rerank={args.topk_rerank}.json"
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
    parser.add_argument("--database_dir", type=str, default="../database/MRAG_corpus_ReT_caption")
    parser.add_argument("--index_dir", type=str, default="../database/MRAG_corpus_ReT_caption/index")
    args = parser.parse_args()
    main(args)
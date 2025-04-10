import os
import numpy as np
from tqdm import tqdm
import csv
import pandas as pd
from PIL import Image
import argparse
import json
from utils import init_dataset, init_encode_model
from retriever import Retriever

def main(args):
    
    os.makedirs(args.retrieval_dir, exist_ok=True)
    
    # init dataset
    dataset = init_dataset(args.dataset_name)

    # init encode model
    encode_model, dim  = init_encode_model(args.model_name_encode)
    
    # vt database
    db = Retriever(args.index_dir, encode_model, dim)

    # extract retriever paths
    results = []
    for (question, question_img, gt_files, choices, gt_ans) in tqdm(dataset.loader()):
        retrieval_paths = db.flow_search(question_img, args.question_dir, filter=0, k=args.topk, topk_rerank=args.topk_rerank)
        results.append(retrieval_paths)
    
    # save results
    output_path = os.path.join(args.retrieval_dir, f"{args.dataset_name}_encoder={args.model_name_encode}_topk={args.topk}_topk_rerank={args.topk_rerank}.json")
    with open(output_path, "w") as f:    
        json.dump(results, f)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_encode", type=str, default="ReT")
    parser.add_argument("--topk_rerank", type=int, default=10)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--question_dir", type=str, default="../dataset/MRAG")
    parser.add_argument("--dataset_dir", type=str, default="../dataset/MRAG_corpus")
    parser.add_argument("--index_dir", type=str, default="../database/MRAG_corpus_ReT_caption/index")
    parser.add_argument("--dataset_name", type=str, default="MRAG")
    parser.add_argument("--results_dir", type=str, default="results_retriever")
    args = parser.parse_args()
    
    main(args)
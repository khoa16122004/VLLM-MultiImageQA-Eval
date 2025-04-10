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
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    # init dataset
    dataset = init_dataset(args.dataset_name)

    # init encode model
    encode_model, dim  = init_encode_model(args.model_name_encode)
    
    # vt database
    map_path = os.path.join(args.index_dir, "map.csv")
    db = Retriever(args.index_dir, encode_model, dim, map_path)

    # extract retriever paths
    results = []
    for (question, question_img, gt_files, choices, gt_ans) in tqdm(dataset.loader()):
        retrieval_paths = db.flow_search(question_img, k=args.topk, topk_rerank=args.topk_rerank)
        results.append(retrieval_paths)
    
    # save results
    output_path = os.path.join(args.results_dir, f"{args.dataset_name}_encoder={args.x}_topk={args.topk}_topk_rerank={args.topk_rerank}.json")
    with open(output_path, "w") as f:    
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_encode", type=str, default="ReT")
    parser.add_argument("--topk_rerank", type=int, default=10)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--index_dir", type=str)
    parser.add_argument("--dataset_name", type=str, default="MRAG")
    parser.add_argument("--results_dir", type=str, default="results_retriever")
    args = parser.parse_args()
    
    main(args)
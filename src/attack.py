
import os
import numpy as np
from tqdm import tqdm
import csv
import pandas as pd
from PIL import Image
import argparse
import json
from utils import init_dataset, init_encode_model
from retriever import Retriever, MultiModal_Retriever
from algorithm import GA

def main(args):
    dataset = init_dataset(args.dataset_name)
    
    # init retriever
    if args.multimodel_retrieval == 0:
        # init encode model
        encode_model, dim  = init_encode_model(args.model_name_encode)
        map_path = os.path.join(args.index_dir, "map.csv")
        db = Retriever(args.index_dir, encode_model, dim, map_path)

    else:
        retrievers = []
        weights = []
        while True:
            model_name = input("Input name of the model")
            if model_name == "q":
                break
            
            encode_model, dim = init_encode_model(model_name)
            index_dir = input("Input the index dir: ")
            map_path = os.path.join(args.index_dir, "map.csv")

            retriever = Retriever(index_dir, encode_model, dim, map_path)
            retrievers.append(retriever)
            w = input(f"Input the weight of {model_name}")
            weights.append(w)
        
        db = MultiModal_Retriever(retrievers, weights)
    
    
    # 
    for (id, question, question_img, gt_files, choices, gt_ans) in tqdm(dataset.loader()):
        retrieval_paths, _ = db.flow_search(img=question_img, question=question, k=args.topk, topk_rerank=args.topk_rerank)
        algorithm = GA(question_img=question_img,
                       epsilon=args.epsilon,
                       retriever=db,
                       gt_paths=gt_files,
                       mutation_rate=args.mutation_rate,
                       max_iteration=args.max_iterations,
                       pop_size=args.pop_size)
        adv_paths = algorithm.solve()
        print(adv_paths)
        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_encode", type=str, default="ReT")
    parser.add_argument("--topk_rerank", type=int, default=1000)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--index_dir", type=str)
    parser.add_argument("--dataset_name", type=str, default="MRAG")
    parser.add_argument("--multimodel_retrieval", type=int, default=0)
    
    # attack argument
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--pop_size", type=int, default=100)
    parser.add_argument("--mutation_rate", type=float, default=0.01)
    args = parser.parse_args()

    main(args)
import os
import numpy as np
from tqdm import tqdm
import faiss
import csv
import pandas as pd
from PIL import Image
import argparse
import ast
from utils import init_encode_model
from retriever import Retriever
def main(args):
    encode_model = init_encode_model(args.model_name_encode)
    retriever = Retriever(args.index_dir, encode_model)
    
    # Extract feature
    retriever.extract(args.dataset_dir, args.database_dir)
    # Indexing
    retriever.create_database(args.database_dir, args.output_index_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_encode", type=str)
    parser.add_argument("--topk_rerank", type=int, default=10)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--question_dir", type=str, default="../dataset/MRAG")
    parser.add_argument("--dataset_dir", type=str, default="../dataset/MRAG_corpus")
    parser.add_argument("--database_dir", type=str)
    parser.add_argument("--output_index_dir", type=str)
    args = parser.parse_args()
    
    main(args)
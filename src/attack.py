import argparse
import numpy as np
from tqdm import tqdm
import random
import torch
from extraction import CreateDatabase
from model import ReTWrapper, MyCLIPWrapper

def main(args):
    if args.model_name_encode == "ReT":
        model_encode = ReTWrapper()
        
    elif args.model_name_encode == "CLIP":
        model_encode = MyCLIPWrapper()
        
    db = CreateDatabase(index_dir=args.database_dir,
                    dataset_dir=args.dataset_dir,
                    model=model_encode,
                    model_name=args.model_name_encode,
                    model_filter=None,
                    caption_model=None)

    sample_paths = db.flow_search(args.image_index, args.question_dir, filter=0, k=args.topk, topk_rerank=args.topk_rerank)
    print(sample_paths)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_index", type=int, default=0)
    parser.add_argument("--pretrained_caption", type=str, default="llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model_name_caption", type=str, default="llava_qwen")
    parser.add_argument("--model_name_encode", type=str, default="ReT")
    parser.add_argument("--topk_rerank", type=int, default=10)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--sample_id_eval", type=int, default=-1)
    parser.add_argument("--dataset_dir", type=str, default="../dataset/MRAG_corpus")
    parser.add_argument("--database_dir", type=str, default="../database/MRAG_corpus_ReT_caption")
    parser.add_argument("--index_dir", type=str, default="../database/MRAG_corpus_ReT_caption/index")
    
    args = parser.parse_args()
    
    main(args)
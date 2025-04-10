import argparse
import numpy as np
from tqdm import tqdm
import random
import torch
from extraction import CreateDatabase
from model import ReTWrapper, MyCLIPWrapper
import os
from PIL import Image
import warnings
warnings.simplefilter("ignore")


def benchmark(pertubation_examples, fea_retri, retri_paths, db, clip_model):
    
    # fea_retri: 5 * dim
    
    # convert to PIL Image
    pil_pertubation_examples = [Image.fromarray(np.uint8(pertubation_example * 255)) for pertubation_example in pertubation_examples]
    # print(pil_pertubation_examples)
    # input()
    # CLIP sim
    fea_pertubation_examples = clip_model.visual_encode_batch(pil_pertubation_examples) # pop_size * dim
    print("Feature perubtation shape: ", fea_pertubation_examples.shape)
    sim_matrix = fea_pertubation_examples @ fea_retri.T # pop_size x 5
    print("Sim_matrix: ", sim_matrix.shape)
    sim_scores = sim_matrix.mean(dim=1) # pop_size
    print("Sim scores: ", sim_scores.shape)
    print("CLIP sim score: ", sim_scores)
    # retrieval score
    retri_scores = []
    pertbuation_retri_paths = db.batch_search(pil_pertubation_examples, k=5, topk_rerank=10) # pop_size x top_k
    for pertubation_retri in pertbuation_retri_paths: # top_k
        intersection = set(pertubation_retri).intersection(retri_paths)
        retri_scores.append(len(intersection) / len(pertubation_retri))
    
    print("Retri_scores: ", retri_scores)
    

def attack(img, retrived_paths, db, clip_model, args):
    img_np = np.array(img)
    img_np = img_np.astype('float32') / 255.0 # img_np: [0,1]

    retrived_imgs = [Image.open(path).convert("RGB") for path in retrived_paths]
    fea_retrived = clip_model.visual_encode_batch(retrived_imgs)

    
    pertubation_examples = img_np + np.random.rand(*img_np.shape) * args.epsilon
    fitness = benchmark(pertubation_examples, fea_retrived, retrived_paths, db, clip_model)
    return
    


def main(args):
    if args.model_name_encode == "ReT":
        model_encode = ReTWrapper()
        
    elif args.model_name_encode == "CLIP":
        model_encode = MyCLIPWrapper()
    
    clip_model = MyCLIPWrapper()
    
    db = CreateDatabase(index_dir=args.index_dir,
                        dataset_dir=args.dataset_dir,
                        model=model_encode,
                        model_name=args.model_name_encode,
                        model_filter=None,
                        caption_model=None)
    
    img_path = os.path.join(args.question_dir, str(args.image_index), "question_img.png")
    img = Image.open(img_path).convert("RGB")
    
    sample_paths = db.flow_search(img, args.question_dir, filter=0, k=args.topk, topk_rerank=args.topk_rerank)
    sample_paths = [os.path.join(args.dataset_dir, path) for path in sample_paths]
    sample_adv_paths = attack(img, sample_paths, db, clip_model, args)
    # print(sample_paths)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_index", type=int, default=0)
    parser.add_argument("--model_name_caption", type=str, default="llava_qwen")
    parser.add_argument("--model_name_encode", type=str, default="ReT")
    parser.add_argument("--topk_rerank", type=int, default=10)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--dataset_dir", type=str, default="../dataset/MRAG_corpus")
    parser.add_argument("--question_dir", type=str, default="../dataset/MRAG")
    parser.add_argument("--index_dir", type=str, default="../database/MRAG_corpus_ReT_caption/index")
    
    # attack
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--max_evaluations", type=int, default=1000)
    parser.add_argument("--pop_size", type=int, default=100)
    args = parser.parse_args()
    
    main(args)
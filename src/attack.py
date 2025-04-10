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
    pil_pertubation_examples = [Image.fromarray(np.uint8(pertubation_example * 255)) for pertubation_example in pertubation_examples]
    fea_pertubation_examples = clip_model.visual_encode_batch(pil_pertubation_examples) # pop_size * dim
    sim_matrix = fea_pertubation_examples @ fea_retri.T # pop_size x 5
    sim_scores = sim_matrix.mean(axis=1) # pop_size
    retri_scores = []
    pertbuation_retri_paths = db.batch_search(pil_pertubation_examples, k=5, topk_rerank=10) # pop_size x top_k
    for pertubation_retri in pertbuation_retri_paths: # top_k
        intersection = set(pertubation_retri).intersection(retri_paths)
        retri_scores.append(len(intersection) / len(pertubation_retri) * 100) 
    retri_scores = np.array(retri_scores)
    print("retri_scores: ", retri_scores.min())
    # print("sim: ", sim_scores.min())

    return 0.5 * retri_scores + 0.5 * sim_scores     

def attack(img, retrived_names, db, clip_model, args):
    img_np = np.array(img).astype('float32') / 255.0  # [H, W, C]
    h, w, c = img_np.shape

    retrived_imgs = [Image.open(os.path.join(args.dataset_dir, path)).convert("RGB") for path in retrived_names]
    fea_retrived = clip_model.visual_encode_batch(retrived_imgs)
    benchmark(np.array([img_np]), fea_retrived, retrived_names, db, clip_model)
    perturbations = np.random.rand(args.pop_size, h, w, c) * 2 * args.epsilon - args.epsilon

    num_evaluations = 0

    while num_evaluations < args.max_evaluations:
        adv_images = np.clip(img_np + perturbations, 0, 1)

        fitness = benchmark(adv_images, fea_retrived, retrived_names, db, clip_model)
        num_evaluations += args.pop_size
        # print("fitness list: ", fitness)

        elite_idxs = np.argsort(fitness)[:args.num_elites]
        elites = perturbations[elite_idxs]
        print("Fintess: ", fitness[elite_idxs[0]])

        new_perturbations = []
        for _ in range(args.pop_size):
            parent = elites[np.random.randint(len(elites))]
            noise = np.random.normal(scale=args.mutation_std, size=(h, w, c))
            child = np.clip(parent + noise, -args.epsilon, args.epsilon)
            new_perturbations.append(child)

        perturbations = np.array(new_perturbations)

    final_adv_images = np.clip(img_np + perturbations, 0, 1)
    final_fitness = benchmark(final_adv_images, fea_retrived, retrived_names, db, clip_model)
    best_idx = np.argmin(final_fitness)
    best_perturbation = perturbations[best_idx]

    img_np_adv = (np.clip(img_np + best_perturbation, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np_adv)
    return pil_img



    


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
    
    sample_names = db.flow_search(img, args.question_dir, filter=0, k=args.topk, topk_rerank=args.topk_rerank)
    # sample_paths = [os.path.join(args.dataset_dir, path) for path in sample_names]
    adv_img = attack(img, sample_names, db, clip_model, args)
    adv_img.save("test.png")
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
    parser.add_argument("--mutation_std", type=float, default=0.01)
    parser.add_argument("--num_elites", type=int, default=20)
    args = parser.parse_args()
    
    main(args)
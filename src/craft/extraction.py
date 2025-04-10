import os
import numpy as np
from model import MyCLIPWrapper, ReTWrapper
from tqdm import tqdm
import faiss
import csv
import pandas as pd
from PIL import Image
import argparse
import ast

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

class CreateDatabase:
    def __init__(self, index_dir, dataset_dir, model, model_name, model_filter, caption_model=None):
        '''
            Args: 
                dir: Folder dir of N samples
                
                output_dir: Folder dir of output
                - contain N numpy files, each file is 6 numpy arrays

                model: Visual Feature extraction model
        '''
        

        self.model = model
        self.model_name = model_name
        self.caption_model = caption_model
        self.index_dir = index_dir
        self.dataset_dir = dataset_dir
        self.model_filter = model_filter
        
        if model_name == "ReT":
            self.d = 4096
        if model_name == "CLIP":
            self.d = 512
        
        
    def extract(self, output_dir):
        '''
            Extracts features from all N samples, and save in output_dir folder 
        '''
        os.makedirs(output_dir, exist_ok=True)
        
        for file_name in tqdm(os.listdir(self.dataset_dir)):
            if "input" in file_name:
                continue
            file_path = os.path.join(self.dataset_dir, file_name)
            caption = None
            caption_prompt = "Describe the image in great detail, mentioning every visible element, their appearance, location, and how they interact in the scene. <image>"
            if self.caption_model is not None:
                caption = self.caption_model.inference(caption_prompt, [Image.open(file_path).convert("RGB")])[0]
                
                
            if self.model_name == "CLIP":
                vec = self.model.visual_encode(file_path)
            elif self.model_name == "ReT":
                vec = self.model.encode_multimodal(file_path, caption)
            
            np.save(os.path.join(output_dir, f"{file_name}.npy"), vec)
                
        
    def create_database(self, database_dir, output_dir, csv_file='map.csv', batch_size=2000):
        '''
        Create the FAISS index by adding vectors from saved numpy files in batches and split into multiple index files.
        '''
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, csv_file), mode='w', newline='') as csvfile:
            fieldnames = ['index', 'batch_idx', 'img_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            index_id = 0 
            current_index = faiss.IndexFlatL2(self.d)
            print("Starting to create indexing...")
            # print("Estimated total index: ", 1 + self.number_vectors // batch_size)
            batch_retrieval_vectors = []
            all_paths = []
            total_vectors_added = 0

            
            # for k, sample_id in tqdm(enumerate(sorted(os.listdir(database_dir)))):
            for npy_file in tqdm(sorted(os.listdir(database_dir))):
                if not npy_file.endswith(".npy"):
                    continue
                retrieval_vectors = np.load(os.path.join(database_dir, npy_file))
                if self.model_name == "ReT":
                    retrieval_vectors = retrieval_vectors.flatten()
                
                all_paths.append(npy_file.split(".npy")[0])
                batch_retrieval_vectors.append(retrieval_vectors)
                
                if len(np.vstack(batch_retrieval_vectors)) >= batch_size:
                    print(f"Adding batch to index... {index_id}")
                    batch_vectors = np.vstack(batch_retrieval_vectors)
                    current_index.add(batch_vectors.astype('float32'))
                    for i in range(len(batch_retrieval_vectors)):
                        writer.writerow({
                            'index': i,
                            'batch_idx': index_id,
                            'img_path': all_paths[i]
                        })

                    faiss.write_index(current_index, os.path.join(output_dir, f"{index_id}.index"))

                    index_id += 1
                    current_index = faiss.IndexFlatL2(self.d)
                    total_vectors_added += len(batch_vectors)
                    batch_retrieval_vectors = []
                    all_paths = []

            if batch_retrieval_vectors:
                print(f"Adding batch to index... {index_id}")
                batch_vectors = np.vstack(batch_retrieval_vectors)
                current_index.add(batch_vectors.astype('float32'))

                for i in range(len(batch_retrieval_vectors)):
                    writer.writerow({
                        'index':  i,
                        'batch_idx': index_id,
                        'img_path': all_paths[i]
                    })

                faiss.write_index(current_index, os.path.join(output_dir, f"{index_id}.index"))
                total_vectors_added += len(batch_vectors)

            print(f"Database created successfully with multiple indexes with total {total_vectors_added} vectors")
                    
    def search_with_reranking(self, query_vector, k=10, top_rerank=50):
        top_indices, top_batches, top_vectors, df = self.search(query_vector, top_rerank, self.d) # 50 vector
        expanded_query = np.mean(top_vectors, axis=0).astype('float32')
        all_distances = np.linalg.norm(top_vectors - expanded_query.reshape(1, -1), axis=1)  # Euclidean

        reranked_idx = np.argsort(all_distances)[:k]

        final_indices = top_indices[reranked_idx]
        final_batches = top_batches[reranked_idx]

        query_df = pd.DataFrame({
            'index': final_indices,
            'batch_idx': final_batches
        })
        merged = pd.merge(query_df, df, on=['index', 'batch_idx'], how='inner')
        sample_paths = merged['img_path'].to_numpy()
        return sample_paths
    
    def batch_search(self, pil_pertubation_examples, k=5, topk_rerank=10):
        query_vectors = self.model.visual_batch_encode(pil_pertubation_examples)
        
        final_paths = []
        for vector in query_vectors:
            sample_paths = self.search_with_reranking(vector, k, topk_rerank).tolist() 
            final_paths.append(sample_paths)
            
        return final_paths
    
    def search(self, query_vector, top_rerank=50, k=5):
        query_vector = query_vector.astype('float32') 
        all_distances = []
        all_indices = []
        all_batch_index = []
        all_vectors = []

        df = pd.read_csv(os.path.join(self.index_dir, 'map.csv'))

        for i in range(len(os.listdir(self.index_dir)) - 1):
            index_path = os.path.join(self.index_dir, f"{i}.index")
            index = faiss.read_index(index_path)
            distances, indices = index.search(query_vector.reshape(1, -1), top_rerank)

            for j in range(len(indices[0])):
                vector = index.reconstruct(int(indices[0][j]))
                all_distances.append(distances[0][j])
                all_indices.append(indices[0][j])
                all_batch_index.append(i)
                all_vectors.append(vector)

        all_distances = np.array(all_distances)
        all_indices = np.array(all_indices)
        all_batch_index = np.array(all_batch_index)
        all_vectors = np.array(all_vectors)

        sorted_idx = np.argsort(all_distances)
        sorted_idx = sorted_idx[:k]

        top_indices = all_indices[sorted_idx]
        top_batches = all_batch_index[sorted_idx]
        top_vectors = all_vectors[sorted_idx]
        
        return top_indices, top_batches, top_vectors, df 
    
    def flow_search(self, img, question_dir, question=None, filter=0, k=10, topk_rerank=10):
        # img_path = os.path.join(question_dir, str(image_index), "question_img.png")
        if self.model_name == "CLIP":
            img_vector = self.model.visual_encode(img)
        elif self.model_name == "ReT":
            # intruction = "Retrieve documents that provide an answer to the question alongside the image: "
            # full_question = intruction + question
            # img_vector = self.model.encode_multimodal(img_path, full_question).flatten()
            img_vector = self.model.encode_multimodal(img) 
        
        if filter == 1:
            sample_paths = self.search_with_reranking(img_vector, 20, topk_rerank).tolist() 
            print("Original_paths: ", sample_paths)
            sample_paths = self.filter(sample_paths, img, self.model, self.dataset_dir)[:k]
            print("Filter Paths: ", sample_paths)
        else:
            sample_paths = self.search_with_reranking(img_vector, k, topk_rerank).tolist() 

        return sample_paths

    def filter(self, img_paths, question_img, main_object="object", dataset_dir="../dataset/MRAG_corpus"):
        images_placeholder = "<image>" * len(img_paths)
        prompt = f"We provide a reference image <image> followed by several candidate images {images_placeholder.strip()}. For each candidate image, determine whether it depicts the same type of {main_object} as the reference image. Return a list with the same length as the number of candidate images: use 1 if it matches the reference image, and 0 otherwise. The output format should be like [1, 0, 1, 0]."

        img_files = [Image.open(question_img).convert('RGB')] + [Image.open(os.path.join(dataset_dir, img_path)).convert('RGB') for img_path in img_paths]
        answer = self.model_filter.inference(prompt, img_files)[0]  # e.g., [1, 0, 1, 0]
        takes = ast.literal_eval(answer)
        filter_paths = []
        for file, take in zip(img_paths, takes):
            if take == 1:
                filter_paths.append(file)
                
        print("Takes: ", takes)
        return filter_paths
        
    

    
def init_caption_model(args):
    special_token = None
    if "llava" in args.model_name_caption:
        from models.llava_ import LLava
        image_token = "<image>"
        model = LLava(args.pretrained_caption, args.model_name_caption)

    elif "openflamingo" in args.model_name_caption:
        from models.openflamingo_ import OpenFlamingo
        image_token = "<image>"
        special_token = "<|endofchunk|>"

        model = OpenFlamingo(args.pretrained_caption)
    
    elif "mantis" in args.model_name_caption:
        from models.mantis_ import Mantis
        image_token = "<image>"
        model = Mantis(args.pretrained_caption)
        
    elif "deepseek" in args.model_name_caption:
        from models.deepseek_ import DeepSeek
        image_token = "<image_placeholder>"
        model = DeepSeek(args.pretrained_caption)
    
    return model, image_token, special_token


    
def main(args):
    if args.model_name_encode == "ReT":
        model_encode = ReTWrapper()
    elif args.model_name_encode == "CLIP":
        model_encode = MyCLIPWrapper()
        
    # caption_model, image_token, special_token = init_caption_model(args)
    
    db = CreateDatabase(index_dir=args.database_dir,
                        dataset_dir=args.dataset_dir,
                        model=model_encode,
                        model_name=args.model_name_encode,
                        model_filter=None,
                        caption_model=None)
    
    if args.action == "indexing":
        db.extract(args.database_dir)       
        db.create_database(args.database_dir, output_dir=args.index_dir)
    
    elif args.action == "search": 
        while True:
            image_index = int(input("Input sampe index: "))
            
            if image_index == -1:
                break
            
            sample_indices = db.flow_search(args.index_dir, args.dataset_dir, image_index)
            print("Results retreval: ", sample_indices)
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_caption", type=str, default="llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model_name_caption", type=str, default="llava_qwen")
    parser.add_argument("--model_name_encode", type=str, default="ReT")
    parser.add_argument("--topk_rerank", type=int, default=10)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--sample_id_eval", type=int, default=-1)
    parser.add_argument("--using_retrieval", type=int, default=1)
    parser.add_argument("--question_dir", type=str, default="../dataset/MRAG")
    parser.add_argument("--dataset_dir", type=str, default="../dataset/MRAG_corpus")
    parser.add_argument("--database_dir", type=str, default="../database/MRAG_corpus_ReT_caption")
    parser.add_argument("--index_dir", type=str, default="../database/MRAG_corpus_ReT_caption/index")
    parser.add_argument("--action", type=str)
    args = parser.parse_args()
    
    main(args)
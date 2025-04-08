import os
import numpy as np
from model import MyCLIPWrapper, ReTWrapper
from tqdm import tqdm
import faiss
import csv
import pandas as pd
from PIL import Image

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
    def __init__(self, model, model_name):
        '''
            Args: 
                dir: Folder dir of N samples
                
                output_dir: Folder dir of output
                - contain N numpy files, each file is 6 numpy arrays

                model: Visual Feature extraction model
        '''
        

        self.model = model
        self.model_name = model_name
        if model_name == "ReT":
            self.d = 4096
        if model_name == "CLIP":
            self.d = 512
        
        
    def extract(self, dir, output_dir):
        '''
            Extracts features from all N samples, and save in output_dir folder 
        '''
        os.makedirs(output_dir, exist_ok=True)
        
        for file_name in tqdm(os.listdir(dir)):
            if "input" in file_name:
                continue
            
            file_path = os.path.join(dir, file_name)
            if self.model_name == "CLIP":
                vec = self.model.visual_encode(file_path)
            elif self.model_name == "ReT":
                vec = self.model.encode_multimodal(file_path)
            
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
            print("Starting to create database...")
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
                    
    def search_with_reranking(self, index_dir, query_vector, k=10, top_rerank=50):
        top_indices, top_batches, top_vectors, df = self.search(index_dir, query_vector, top_rerank, self.d) # 50 vector
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
            
    
    def search(self, index_dir, query_vector, top_rerank=50, k=5):
        query_vector = query_vector.astype('float32') 
        all_distances = []
        all_indices = []
        all_batch_index = []
        all_vectors = []

        df = pd.read_csv(os.path.join(index_dir, 'map.csv'))

        for i in range(len(os.listdir(index_dir)) - 1):
            index_path = os.path.join(index_dir, f"{i}.index")
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

        # query_df = pd.DataFrame({
        #     'index': top_indices,
        #     'batch_idx': top_batches
        # })

        # merged = pd.merge(query_df, df, on=['index', 'batch_idx'], how='inner')

        return top_indices, top_batches, top_vectors, df 
    
    def flow_search(self, index_dir, dataset_dir, image_index, k=10, topk_rerank=10):
        img_path = os.path.join(dataset_dir, str(image_index), "question_img.png")
        if self.model_name == "CLIP":
            img_vector = self.model.visual_encode(img_path)
        elif self.model_name == "ReT":
            img_vector = self.model.encode_multimodal(img_path).flatten()
            
        sample_indices = self.search_with_reranking(index_dir, img_vector, k, topk_rerank)
        
        return sample_indices


class MultiModelSearch:
    def __init__(self, dbs): # list of db
        self.dbs = dbs
        self.weights = [1/len(dbs)] * len(dbs)
        
    def search_with_reranking(self, index_dir, query_vector, k=10, top_rerank=50):
        for db in dbs:
            sample_paths = db.search_with_reranking(index_dir, query_vector, k, top_rerank)
            return sample_paths


    
if __name__ == "__main__":
    # model = MyCLIPWrapper()
    model_encode = ReTWrapper()
    db = CreateDatabase(model=model_encode, model_name="ReT")
    
    question_dir = "../dataset/MRAG"
    dataset_dir = "../dataset/MRAG_corpus"
    database_dir = "../database/MRAG_corpus_ReT_new"
    index_dir = "../database/MRAG_corpus_ReT_new/index"
    
    # db.extract(dataset_dir, database_dir)       
    db.create_database(database_dir, output_dir=index_dir)
    while True:
        image_index = int(input("Input sampe index: "))
        
        if image_index == -1:
            break
        
        sample_indices = db.flow_search(index_dir, question_dir, image_index)
        print("Results retreval: ", sample_indices)

import os
from tqdm import tqdm
import numpy as np
import faiss
import pandas as pd
import csv
from PIL import Image

class Retriever:
    def __init__(self, index_dir, encode_model, dim, map_path=None):
        self.index_dir = index_dir
        self.encode_model = encode_model
        self.map_path = map_path
        self.d = dim
        
    def extract_db(self, dataset_dir, output_dir, caption_dir=None):
        
        """
            Args:
                dataset_dir: Folder dir of N Image samples
                output_dir: Folder dir of output indexing file
                
            Return:
                None
        """
        os.makedirs(output_dir, exist_ok=True)
        print("Extract Feature Proccess ...")
        for img_name in tqdm(os.listdir(dataset_dir)):
            img_path = os.path.join(dataset_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            
            # if caption
            caption = ""
            if caption_dir:
                caption_path = os.path.join(caption_dir, img_name + ".txt")
                with open(caption_path, "r") as f:
                    caption = f.readline().strip()
                    
                
            vec = self.encode_model.visual_encode(img, caption)
            if self.encode_model.name == "CLIP":
                np.save(os.path.join(output_dir, f"{img_name}_0.npy"), vec)
            
            elif self.encode_model.name == "ReT":
                np.save(os.path.join(output_dir, f"{img_name}_0.npy"), vec[0])

        print("Done Extract Feature")        
    
    
    def create_database(self, vectors_dir, output_dir, csv_file='map.csv', batch_size=9000):
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
            for npy_file in tqdm(sorted(os.listdir(vectors_dir))):
                if not npy_file.endswith(".npy"):
                    continue
                retrieval_vectors = np.load(os.path.join(vectors_dir, npy_file)) # 1 x dim
                
                
                original_name = "_".join(npy_file.split("_")[:-1])
                all_paths.append(original_name)
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
    
    def map(self, top_indies, top_batchs):
        df = pd.read_csv(self.map_path)
        merged = pd.merge(pd.DataFrame({'index': top_indies, 'batch_idx': top_batchs}), 
                          df, on=['index', 'batch_idx'], 
                          how='inner')
        
        img_paths = merged['img_path'].to_numpy().tolist()
        return img_paths
    
    def search(self, query_vector, top_rerank=50, k=5):
        '''
            Args:
                query vector: encoded from encoder
            
            Return:
                image paths: List[img_path (str)]
        
        '''
        
        # if 1D vector
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        
        query_vector = query_vector.astype('float32') 
        all_distances = [[] * len(query_vector)]
        print(len(all_distances))
        all_indices = [[] * len(query_vector)]
        all_batch_index = [[] * len(query_vector)]

        for i in range(len(os.listdir(self.index_dir)) - 1):
            index_path = os.path.join(self.index_dir, f"{i}.index")
            index = faiss.read_index(index_path)
            distances, indices = index.search(query_vector, top_rerank) # B x top_rerank

            for j in range(len(query_vector)):
                all_distances[j].extend(distances[0])
                all_indices[j].extend(indices[0])
                all_batch_index[j].extend([i] * len(indices[0]))
            
        all_distances = np.array(all_distances)
        all_indices = np.array(all_indices)
        all_batch_index = np.array(all_batch_index)

        sorted_idx = np.argsort(all_distances, axis=0)
        print(sorted_idx.shape)
        sorted_idx = sorted_idx[:, :k]

        top_indices = all_indices[sorted_idx]
        top_batches = all_batch_index[sorted_idx]
        top_distance = all_distances[sorted_idx]
        
        print(top_indices.shape)
        
        return top_distance, top_indices, top_batches
    
    def flow_search(self, img, question=None, k=10, topk_rerank=10):
        img_vector = self.encode_model.visual_encode(img, question) # 32 x d            
        top_distance, top_indices, top_batches = self.search(img_vector, k, topk_rerank)
        img_paths = self.map(top_indices, top_batches)
        return img_paths, top_distance

class MultiModal_Retriever:
    def __init__(self, retriever_list, weights):
        self.retriever_list = retriever_list
        
    def flow_search(self, img, question=None, k=10, topk_rerank=10, topk_each_model=100):
        results = {}
        all_paths = set()
        for retriever in self.retriever_list:
            img_paths, distances = retriever.flow_search(img, question, topk_each_model, 100)
            results[retriever.__class__.__name__] = {
                path: distance for path, distance in zip(img_paths, distances)
            }
            all_paths.update(img_paths)
        
        final_results = []   
        for path in all_paths:
            avg_score = 0.0
            for w, retri_name in zip(w, results):
                avg_score += w * results[retri_name].get(path, 0.0)
            final_results.append((path, avg_score))
            
        final_results = sorted(final_results, key=lambda x: x[1])
        paths = [final_result[0] for final_result in final_results][:k]
        return paths
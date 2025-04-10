import os
from tqdm import tqdm
import numpy as np
import faiss
import pandas as pd
import csv

class Retriever:
    def __init__(self, index_dir, encode_model, dim, map_path=None):
        self.index_dir = index_dir
        self.encode_model = encode_model
        self.map_path = map_path
        self.d = dim
        
    def extract_db(self, dataset_dir, output_dir):
        
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
            vec = self.encode_model.visual_encode(img_path, "")
            np.save(os.path.join(output_dir, f"{img_name}.npy"), vec)
        print("Done Extract Feature")        
    
    
    def create_database(self, vectors_dir, output_dir, csv_file='map.csv', batch_size=2000):
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
                retrieval_vectors = np.load(os.path.join(vectors_dir, npy_file))
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
        
        '''
            Args:
                query vector: encoded from encoder
            
            Return:
                image paths: List[img_path (str)]
        
        '''
        
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
        img_paths = merged['img_path'].to_numpy()
        return img_paths
    
    
    def search(self, query_vector, top_rerank=50, k=5):
        '''
            Args:
                query vector: encoded from encoder
            
            Return:
                image paths: List[img_path (str)]
        
        '''
        
        query_vector = query_vector.astype('float32') 
        all_distances = []
        all_indices = []
        all_batch_index = []
        all_vectors = []

        df = pd.read_csv(self.map_path)

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
    
    
    def batch_search(self, pil_pertubation_examples, k=5, topk_rerank=10):
        
        """
            Args:
                pil_pertubation_examples: List[pil_image]
            
            Return:
                image paths: List[List[img_path (str)]]
        """
        
        query_vectors = self.encode_model.visual_batch_encode(pil_pertubation_examples)
        
        final_paths = []
        for vector in query_vectors:
            sample_paths = self.search_with_reranking(vector, k, topk_rerank).tolist() 
            final_paths.append(sample_paths)
            
        return final_paths
    
    
    def flow_search(self, img, k=10, topk_rerank=10):
        img_vector = self.encode_model.visual_encode(img)
        img_paths = self.search_with_reranking(img_vector, k, topk_rerank).tolist() 
        return img_paths
    
    
        
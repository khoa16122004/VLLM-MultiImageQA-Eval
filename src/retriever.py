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
                for i in range(len(vec)):
                    np.save(os.path.join(output_dir, f"{img_name}_{i}.npy"), vec[i])

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
    
    
    def search_with_reranking(self, query_vector, k=10, top_rerank=50):
        
        '''
            Args:
                query vector: encoded from encoder
            
            Return:
                image paths: List[img_path (str)]
        
        '''
        
        top_distances, top_indices, top_batches, top_vectors, df = self.search(query_vector, top_rerank, self.d) # 50 vector
        expanded_query = np.mean(top_vectors, axis=0).astype('float32')
        all_distances = np.linalg.norm(top_vectors - expanded_query.reshape(1, -1), axis=1)  # Euclidean

        reranked_idx = np.argsort(all_distances)[:k]

        final_indices = top_indices[reranked_idx]
        final_batches = top_batches[reranked_idx]
        final_distances = top_distances[reranked_idx]

        query_df = pd.DataFrame({
            'index': final_indices,
            'batch_idx': final_batches
        })
        merged = pd.merge(query_df, df, on=['index', 'batch_idx'], how='inner')
        img_paths = merged['img_path'].to_numpy().tolist()
        return img_paths, final_distances
    
    
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
        top_distance = all_distances[sorted_idx]
        
        return top_distance, top_indices, top_batches, top_vectors, df 
    
    
    def ReT_search(self, query_matrix, k=5, topk_rerank=100):
        paths_count_dict = {}
        paths_distance_dict = {}

        all_distances = []
        all_indices = []
        all_batch_index = []

        df = pd.read_csv(self.map_path)

        for i in range(len(os.listdir(self.index_dir)) - 1):
            index_path = os.path.join(self.index_dir, f"{i}.index")
            index = faiss.read_index(index_path)

            distances, indices = index.search(query_matrix, topk_rerank)  # [batch_size, topk_rerank]

            # Flatten và lưu lại
            all_distances.extend(distances.flatten())
            all_indices.extend(indices.flatten())
            all_batch_index.extend([i] * (distances.shape[0] * distances.shape[1]))
        
        
        all_distances = np.array(all_distances)
        all_indices = np.array(all_indices)
        all_batch_index = np.array(all_batch_index)
        sorted_idx = np.argsort(all_distances)
        sorted_idx = sorted_idx[:topk_rerank]

        top_indices = all_indices[sorted_idx]
        top_batches = all_batch_index[sorted_idx]
        top_distance = all_distances[sorted_idx]        
        
        query_df = pd.DataFrame({
            'index': top_indices,
            'batch_idx': top_batches
        })
        merged = pd.merge(query_df, df, on=['index', 'batch_idx'], how='inner')
        img_paths = merged['img_path'].to_numpy().tolist()
        
        for path, dis in zip(img_paths, top_distance):
            paths_count_dict[path] = paths_count_dict.get(path, 0) + 1
            paths_distance_dict[path] = paths_distance_dict.get(path, 0) + dis
        
        scores = [(path ,paths_distance_dict[path] / paths_count_dict[path]) for path in paths_count_dict]
        scores = sorted(scores, key=lambda x: x[1])[:k]
        
        return scores
                
    
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
    
    
    def flow_search(self, img, question=None, k=10, topk_rerank=10):
        img_vector = self.encode_model.visual_encode(img, question) # 32 x d
        if self.encode_model.name == "ReT":
            results = self.ReT_search(img_vector, k, topk_rerank)
            img_paths = [result[0] for result in results]
            distances = [result[1] for result in results]
        elif self.encode_model.name == "CLIP":
            img_paths, distances = self.search_with_reranking(img_vector, k, topk_rerank)
        
        
        return img_paths, distances

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
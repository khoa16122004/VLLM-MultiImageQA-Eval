import os
import numpy as np
from model import MyCLIPWrapper
from tqdm import tqdm
import faiss
import csv
import pandas as pd

class CreateDatabase:
    def __init__(self, model, number_v=6357):
        '''
            Args: 
                dir: Folder dir of N samples
                
                output_dir: Folder dir of output
                - contain N numpy files, each file is 6 numpy arrays

                model: Visual Feature extraction model
        '''
        

        self.model = model
        self.number_vectors = number_v
        
        
    def extract(self, dir, output_dir):
        '''
            Extracts features from all N samples, and save in output_dir folder 
        '''
        os.makedirs(output_dir, exist_ok=True)

        
        for sample_id in tqdm(os.listdir(dir)):
            if not sample_id.endswith(".py"):
                
                sample_dir = os.path.join(dir, sample_id)
                sample_dir_output = os.path.join(output_dir, sample_id)
                os.makedirs(sample_dir_output, exist_ok=True)

                retrieved_vectors = []
                qs_vector = None

                img_paths = []
                for img_name in os.listdir(sample_dir):
                    img_path = os.path.join(sample_dir, img_name)
                    if "gt" in img_name:
                        vec = self.model.visual_encode(img_path)
                        retrieved_vectors.append(vec)
                        img_paths.append(img_path)
                    elif "question" in img_name and img_name.endswith(".png"):
                        qs_vector = self.model.visual_encode(img_path)

                if qs_vector is not None and len(retrieved_vectors) > 0:
                    np.save(os.path.join(sample_dir_output, "question.npy"), qs_vector)
                    np.save(os.path.join(sample_dir_output, "paths.npy"), img_paths)
                    np.save(os.path.join(sample_dir_output, "retrieval.npy"), np.stack(retrieved_vectors))
                
        
    def create_database(self, database_dir, output_dir, d=512, csv_file='map.csv', batch_size=1000):
        '''
        Create the FAISS index by adding vectors from saved numpy files in batches and split into multiple index files.
        '''
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, csv_file), mode='w', newline='') as csvfile:
            fieldnames = ['index', 'sample_id', 'batch_idx', 'img_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            index_id = 0 
            current_index = faiss.IndexFlatL2(d)
            print("Starting to create database...")
            print("Estimated total index: ", 1 + self.number_vectors // batch_size)
            batch_retrieval_vectors = []
            batch_sample_ids = []
            all_paths = []
            total_vectors_added = 0

            
            for k, sample_id in tqdm(enumerate(sorted(os.listdir(database_dir)))):
                if sample_id.endswith(".py") or sample_id == "index":
                    continue

                print(f"Read sample {k} ...")
                sample_dir_input = os.path.join(database_dir, sample_id)
                retrieval_vectors = np.load(os.path.join(sample_dir_input, "retrieval.npy"))
                paths = np.load(os.path.join(sample_dir_input, "paths.npy"))
                all_paths.extend(paths)
                batch_retrieval_vectors.append(retrieval_vectors)
                batch_sample_ids.extend([sample_id] * retrieval_vectors.shape[0])
                
                if len(np.vstack(batch_retrieval_vectors)) >= batch_size:
                    print(f"Adding batch to index... {index_id}")
                    batch_vectors = np.vstack(batch_retrieval_vectors)
                    current_index.add(batch_vectors.astype('float32'))

                    for i, sid in enumerate(batch_sample_ids):
                        writer.writerow({
                            'index': i,
                            'sample_id': sid,
                            'batch_idx': index_id,
                            'img_path': all_paths[i]
                        })

                    faiss.write_index(current_index, os.path.join(output_dir, f"{index_id}.index"))

                    index_id += 1
                    current_index = faiss.IndexFlatL2(d)
                    total_vectors_added += len(batch_vectors)
                    batch_retrieval_vectors = []
                    batch_sample_ids = []
                    all_paths = []

            if batch_retrieval_vectors:
                print(f"Adding batch to index... {index_id}")
                batch_vectors = np.vstack(batch_retrieval_vectors)
                current_index.add(batch_vectors.astype('float32'))

                for i, sid in enumerate(batch_sample_ids):
                    writer.writerow({
                        'index':  i,
                        'sample_id': sid,
                        'batch_idx': index_id,
                        'img_path': all_paths[i]
                    })

                faiss.write_index(current_index, os.path.join(output_dir, f"{index_id}.index"))
                total_vectors_added += len(batch_vectors)

            print(f"Database created successfully with multiple indexes with total {total_vectors_added} vectors")
                    
    def search_with_reranking(self, index_dir, query_vector, k=10, top_rerank=50, d=512):
        top_indices, top_batches, top_vectors, df, sample_indices = self.search(index_dir, query_vector, top_rerank, d) # 50 vector

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
        sample_indices = merged['sample_id'].to_numpy()
        return sample_paths, sample_indices
            
    
    def search(self, index_dir, query_vector, top_rerank=50, d=512, k=5):
        query_vector = query_vector.astype('float32') 
        all_distances = []
        all_indices = []
        all_batch_index = []
        all_vectors = []
        
        
        df = pd.read_csv(os.path.join(index_dir, 'map.csv'))
        for i in range(len(os.listdir(index_dir)) - 1):
            index_path = os.path.join(index_dir, f"{i}.index")
            # print(f"Searching in {index_path} ...")
            
            index = faiss.read_index(index_path)
            distances, indices = index.search(query_vector.reshape(1, -1), top_rerank)  
            
            filtered_idx = [indices[0][0]]
            filtered_dist = [distances[0][0]]
            filtered_vectors = [index.reconstruct(int(indices[0][0]))]

            for j in range(1, top_rerank):
                if not np.isclose(distances[0][j], filtered_dist[-1], atol=1e-4):
                    filtered_idx.append(indices[0][j])
                    filtered_dist.append(distances[0][j])
                    filtered_vectors.append(index.reconstruct(int(indices[0][j])))

            all_distances.append(np.array(filtered_dist).reshape(1, -1))
            all_indices.append(np.array(filtered_idx).reshape(1, -1))
            all_batch_index.append([i] * len(filtered_idx))
            all_vectors.extend(filtered_vectors)
            
        
        all_distances = np.hstack(all_distances) # 150,
        all_indices = np.hstack(all_indices) # 150,
        all_batch_index = np.hstack(all_batch_index) # 150, 
        all_vectors = np.vstack(all_vectors)
        
        top_idx = np.argsort(all_distances[0])[:top_rerank]
        top_distances = all_distances[0][top_idx]
        top_indices = all_indices[0][top_idx]
        top_batches = all_batch_index[top_idx]
        top_vectors = all_vectors[top_idx]

        
        query_df = pd.DataFrame({
            'index': top_indices[:k],
            'batch_idx': top_batches[:k]
            })
        
        merged = pd.merge(query_df, df, on=['index', 'batch_idx'], how='inner')
        sample_indices = merged['img_path'].to_numpy()
        
        # print("Before reranking: ", sample_indices)
        return top_indices, top_batches, top_vectors, df, sample_indices
    
    def flow_search(self, index_dir, dataset_dir, image_index, k=10, topk_rerank=10, d=512):
        img_path = os.path.join(dataset_dir, str(image_index), "question_img.png")
        img_vector = self.model.visual_encode(img_path)
        sample_indices = self.search_with_reranking(index_dir, img_vector, k, topk_rerank, d)
        
        return sample_indices
        
if __name__ == "__main__":
    model = MyCLIPWrapper()
    db = CreateDatabase(model=model)
    
    dataset_dir = "../dataset/MRAG"
    database_dir = "../database/MRAG_CLIP"
    # db.extract(dataset_dir, database_dir)       
    # db.create_database(database_dir, output_dir="../database/MRAG/index")

    index_dir = "../database/MRAG_CLIP/index"
    while True:
        image_index = int(input("Input sampe index: "))
        
        if image_index == -1:
            break
        
        sample_indices = db.flow_search(index_dir, dataset_dir, image_index)
        print("Results retreval: ", sample_indices)

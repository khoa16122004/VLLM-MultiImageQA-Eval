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

                for img_name in os.listdir(sample_dir):
                    img_path = os.path.join(sample_dir, img_name)
                    if "gt" in img_name:
                        vec = self.model.visual_encode(img_path)
                        retrieved_vectors.append(vec)
                    elif "question" in img_name and img_name.endswith(".png"):
                        qs_vector = self.model.visual_encode(img_path)

                if qs_vector is not None and len(retrieved_vectors) > 0:
                    np.save(os.path.join(sample_dir_output, "question.npy"), qs_vector)
                    np.save(os.path.join(sample_dir_output, "retrieval.npy"), np.stack(retrieved_vectors))
                
        
    def create_database(self, database_dir, output_dir, d=512, csv_file='map.csv', batch_size=1000):
        '''
        Create the FAISS index by adding vectors from saved numpy files in batches and split into multiple index files.
        '''
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, csv_file), mode='w', newline='') as csvfile:
            fieldnames = ['index', 'sample_id', 'batch_idx']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            index_id = 0 
            current_index = faiss.IndexFlatL2(d)  # Create a fresh index
            print("Starting to create database...")
            print("Estimated total index: ", 1 + self.number_vectors // batch_size)
            batch_retrieval_vectors = []
            total_vectors_added = 0

            for k, sample_id in tqdm(enumerate(sorted(os.listdir(database_dir)))):
                if sample_id.endswith(".py") or sample_id == "index":
                    continue
                
                print(f"Read sample {k} ...")
                sample_dir_input = os.path.join(database_dir, sample_id)

                retrieval_vectors = np.load(os.path.join(sample_dir_input, "retrieval.npy"))
                batch_retrieval_vectors.append(retrieval_vectors)
                if len(np.vstack(batch_retrieval_vectors)) >= batch_size:
                    print(f"Adding batch to index... {index_id}")
                    batch_retrieval_vectors = np.vstack(batch_retrieval_vectors)
                    current_index.add(batch_retrieval_vectors.astype('float32'))
                    writer.writerow({'index': index_id, 
                                     'sample_id': sample_id,
                                     'batch_idx': index_id})

                    faiss.write_index(current_index, os.path.join(output_dir, f"{index_id}.index"))
                    
                    index_id += 1
                    current_index = faiss.IndexFlatL2(d)
                    total_vectors_added += len(batch_retrieval_vectors) 
                    batch_retrieval_vectors = [] 
                    
            print(f"Adding batch to index... {index_id}")
            batch_retrieval_vectors = np.vstack(batch_retrieval_vectors)
            current_index.add(batch_retrieval_vectors.astype('float32'))
            writer.writerow({'index': index_id, 
                             'sample_id': sample_id,
                             'batch_idx': index_id})
            faiss.write_index(current_index, os.path.join(output_dir, f"{index_id}.index"))
            total_vectors_added += len(batch_retrieval_vectors) 


            print(f"Database created successfully with multiple indexes with total {total_vectors_added} vectors")
            

    def search(self, index_dir, query_vector, k=5, d=512):
        query_vector = query_vector.astype('float32')  # Đảm bảo vector có kiểu float32
        all_distances = []
        all_indices = []
        all_batch_index = []
        
        df = pd.read_csv(os.path.join(index_dir, 'map.csv'))

        
        
        for i in range(os.listdir(index_dir)):
            index_path = os.path.join(index_dir, f"{i}.index")
            print(f"Searching in {index_path} ...")
            
            index = faiss.read_index(index_path)

            distances, indices = index.search(query_vector, k)  
            all_distances.append(distances)
            all_indices.append(indices)
            all_batch_index.append([i]*len(indices))
        
        all_distances = np.hstack(all_distances) # 35,
        all_indices = np.hstack(all_indices) # 35,
        all_batch_index = np.hstack(all_batch_index) # 35, 
        best_indices_from_all = np.argsort(all_distances[0])[:k]
        best_batch_index = all_batch_index[0][best_indices_from_all]
        best_indices = all_indices[0][best_indices_from_all]
        query_df = pd.DataFrame({
            'index': best_indices,
            'batch_idx': best_batch_index
            })
        
        merged = pd.merge(query_df, df, on=['index', 'batch_idx'], how='inner')
        sample_indices = merged['sample_id'].to_numpy()
        return sample_indices

    
    def flow_search(self, index_dir, dataset_dir, image_index, k=5, d=512):
        img_path = os.path.join(dataset_dir, image_index, "question_img.png")
        img_vector = self.model.visual_encode(img_path)
        self.search(index_dir, img_vector, k, d)
        
if __name__ == "__main__":
    model = MyCLIPWrapper()
    db = CreateDatabase(model=model)
    
    dataset_dir = "../dataset/MRAG"
    database_dir = "../database/MRAG"
    # db.create_database(database_dir, output_dir="../database/MRAG/index")
    # db.extract(extract_folder, output_dir)       
    
    index_dir = "../database/MRAG/index"
    while True:
        image_index = int(input("Input sampe index"))
        
        if image_index == -1:
            break
        
        sample_indices = db.flow_search(index_dir, dataset_dir, image_index)
        print("Results retreval: ", sample_indices)

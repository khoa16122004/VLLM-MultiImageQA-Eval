import os
import numpy as np
from model import MyCLIPWrapper
from tqdm import tqdm
import faiss
import csv

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
                if len(np.vstack(batch_retrieval_vectors)) >= batch_size or (self.number_vectors - total_vectors_added) < batch_size:
                    print(f"Adding batch to index... {index_id}")
                    input()
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

            print(f"Database created successfully with multiple indexes with total {total_vectors_added} vectors")
            

    def search(self, index_file, query_vector, k=5, d=512):
        '''
            Search the FAISS index for the top-k closest vectors.
            Args:
                index_file (str): Path to the saved FAISS index.
                query_vector (np.ndarray): Query vector (must be 1-dimensional).
                k (int): Number of nearest neighbors to return.
                d (int): Dimension of the vectors (must match the index).
            Returns:
                distances (np.ndarray): Distances of the top-k results.
                indices (np.ndarray): Indices of the top-k results.
        '''
        index = faiss.read_index(index_file)

        query_vector = query_vector.astype('float32').reshape(1, -1)

        distances, indices = index.search(query_vector, k)

        return distances, indices
    
if __name__ == "__main__":
    model = MyCLIPWrapper()
    db = CreateDatabase(model=model)
    
    # extract_folder = "../dataset/MRAG"
    database_dir = "../database/MRAG"
    db.create_database(database_dir, output_dir="../database/MRAG/index")
    # db.extract(extract_folder, output_dir)        
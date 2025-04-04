import os
import numpy as np
from model import MyCLIPWrapper
from tqdm import tqdm
class CreateDatabase:
    def __init__(self, model):
        '''
            Args: 
                dir: Folder dir of N samples
                
                output_dir: Folder dir of output
                - contain N numpy files, each file is 6 numpy arrays

                model: Visual Feature extraction model
        '''
        

        self.model = model
        
        
    def extract(self, dir, output_dir):
        '''
            Extracts features from all N samples, and save in output_dir folder 
        '''
        os.makedirs(output_dir, exist_ok=True)

        
        for sample_id in tqdm(os.listdir(dir)):
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
            
        
    def create_database(self, output_file):
        pass
    
if __name__ == "__main__":
    model = MyCLIPWrapper()
    db = CreateDatabase(model=model)
    
    extract_folder = "../dataset/MRAG"
    output_dir = "../database/MRAG"
    
    db.extract(extract_folder, output_dir)        
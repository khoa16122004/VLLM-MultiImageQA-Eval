import os
import numpy as np
from model import CLIPModel
class CreateDatabase:
    def __init__(self, dir, output_dir, model):
        '''
            Args: 
                dir: Folder dir of N samples
                
                output_dir: Folder dir of output
                - contain N numpy files, each file is 6 numpy arrays

                model: Visual Feature extraction model
        '''
        
        os.makedirs(output_dir, exist_ok=True)
        self.dir = dir
        self.output_dir = output_dir
        self.model = model
        
        
    def extract(self):
        '''
            Extracts features from all N samples, and save in output_dir folder 
        '''
        for sample_id in os.listdir(self.dir):
            sample_dir = os.path.join(self.dir, sample_id)
            sample_dir_output = os.path.join(self.output_dir, sample_id)
            os.makedirs(sample_dir_output, exist_ok=True)

            retrieved_vectors = []
            qs_vector = None

            for img_name in os.listdir(sample_dir):
                img_path = os.path.join(sample_dir, img_name)
                if "gt" in img_name:
                    vec = self.model(img_path)
                    retrieved_vectors.append(vec)
                elif "question" in img_name:
                    qs_vector = self.model(img_path)

            if qs_vector is not None and len(retrieved_vectors) > 0:
                np.save(os.path.join(sample_dir_output, "question.npy"), qs_vector)
                np.save(os.path.join(sample_dir_output, "retrieval.npy"), np.stack(retrieved_vectors))
            
        
    def create_databae(self):
        pass
    
if __name__ == "__main__":
    model = CLIPModel()
    db = CreateDatabase(dir="samples", output_dir="database", model=model).extract()
    
    extract_folder = "../dataset/MRAG"
    output_dir = "../database/MRAG"
    
    db = CreateDatabase(dir=extract_folder, output_dir=output_dir, model=model).extract()
        
import os
from PIL import Image

class MRAG:
    def __init__(self, dataset_dir="../dataset/MRAG_corpus", question_dir="../dataset/MRAG"):
        self.dataset_dir = dataset_dir
        self.question_dir = question_dir
        
        
    def extract_question(self, id):
        
        """
            Args:
                id: id of sample
            
            Return:
                question: str,
                gt_files: list[Image.Image],
                choices: list[str],
                gt_ans: str 
                question_img, gt_files, choices, gt_ans
        """
        
        gt_files = []
        sample_dir = os.path.join(self.question_dir, id)
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
        return id, question, question_img, gt_files, choices, gt_ans
    
    
    def loader(self):
        for sample_id in os.listdir(self.question_dir):
            if sample_id != "index" and not sample_id.endswith(".py"):
                yield self.extract_question(sample_id)
                
    def take_sample(self, sample_id):
        sample_dir = os.path.join(self.question_dir, sample_id)
        question, question_img, gt_files, choices, gt_ans = self.extract_question(sample_id)
        return question, question_img, gt_files, choices, gt_ans

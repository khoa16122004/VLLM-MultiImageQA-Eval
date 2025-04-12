import numpy as np
from tqdm import tqdm
from PIL import Image

class GA:
    def __init__(self,
                 question_img,
                 question, 
                 epsilon,
                 retriever,
                 gt_paths,
                 k=5,
                 topk_rerank=10,
                 mutation_rate=0.1,
                 max_iteration=1000,
                 pop_size=100):
        
        self.max_iteration = max_iteration
        self.pop_size = pop_size
        self.question_img = question_img.resize((256, 256))
        self.np_question_img = np.array(self.question_img) / 255
        self.epsilon = epsilon
        self.mutation_rate = mutation_rate
        self.retriever = retriever
        self.gt_paths = gt_paths  # dictionary: path -> True/False
        self.question_img = question_img  # assuming retriever.flow_search uses this
        self.question = question
        self.k = k
        self.topk_rerank = topk_rerank
    def fitness(self, P):
        pil_imgs = [Image.fromarray(np.uint8(np.clip((self.np_question_img + p) * 255, 0, 255))) for p in P]
        batch_paths, distances = self.retriever.flow_search(pil_imgs, self.question, self.k * 2, self.topk_rerank)
        # retrieved encode
        fitness_scores = []
        for paths, dis in zip(batch_paths, distances):
            scores = [1 / (dis[i] * (i + 1)) if self.is_gt(paths[i]) else (i + 1) / dis[i] for i in range(self.k * 2)]
            fitness_scores.append(np.sum(scores))
            
            
        # clip sim score
        
        return np.array(fitness_scores), batch_paths, pil_imgs

    def is_gt(self, path):
        return self.gt_paths.get(path, False)

    def tournament_selection(self, fitness_scores, k=4):
        indices = np.random.choice(len(fitness_scores), k, replace=False)
        return indices[np.argmin(fitness_scores[indices])]

    def solve(self):
        w, h, c = self.np_question_img.shape
        P = np.random.normal(loc=0.0, scale=self.epsilon, size=(self.pop_size, w, h, c))
        P = np.clip(P, -self.epsilon, self.epsilon)
        P_fitness, P_batch_paths, P_pil_imgs = self.fitness(P)
        print("P fitness: ", P_fitness)
        
        for _ in tqdm(range(self.max_iteration)):
            parent_indices = np.random.randint(0, self.pop_size, size=(self.pop_size, 2))
            
            
            
            O = P[parent_indices[:, 0]] + P[parent_indices[:, 1]]
            mutation_mask = np.random.rand(self.pop_size, 1, 1, 1) < self.mutation_rate            
            mutation_values = np.random.normal(loc=0.0, scale=self.epsilon, size=(self.pop_size, w, h, c))
            mutation_values = np.clip(P, -self.epsilon, self.epsilon)
            O = np.where(mutation_mask, mutation_values, O)
            O = np.clip(O, -self.epsilon, self.epsilon)

            O_fitness, O_batch_paths, O_pil_imgs = self.fitness(O)

            pool = np.concatenate((P, O), axis=0)
            pool_fitness = np.concatenate((P_fitness, O_fitness), axis=0)
            pool_batch_paths = np.concatenate((P_batch_paths, O_batch_paths), axis=0)
            pool_pil_imgs = P_pil_imgs + O_pil_imgs
            
            selected_P = []
            selected_fitness = []
            selected_batch_paths = []
            selected_pil_imgs = []
            for _ in range(self.pop_size):
                idx = self.tournament_selection(pool_fitness)
                selected_P.append(pool[idx])
                selected_fitness.append(pool_fitness[idx])
                selected_batch_paths.append(pool_batch_paths[idx])
                selected_pil_imgs.append(pool_pil_imgs[idx])
                
            P = np.stack(selected_P)
            P_fitness = np.array(selected_fitness)
            P_batch_paths = np.stack(selected_batch_paths)
            P_pil_imgs = selected_pil_imgs
            
            print("Best Fitness: ", np.min(P_fitness))
            # print("Best Path: ", P_batch_paths[np.argmin(P_fitness)])

        best_idx = np.argmin(P_fitness)
        return P[best_idx], P_fitness[best_idx], P_batch_paths[best_idx], P_pil_imgs[best_idx]

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
                 mutation_rate=0.1,
                 max_iteration=1000,
                 pop_size=100):
        
        self.max_iteration = max_iteration
        self.pop_size = pop_size
        self.question_img = question_img
        self.np_question_img = np.array(question_img)
        self.epsilon = epsilon
        self.mutation_rate = mutation_rate
        self.retriever = retriever
        self.gt_paths = gt_paths  # dictionary: path -> True/False
        self.question_img = question_img  # assuming retriever.flow_search uses this
        self.question = question
    def fitness(self, P):
        pil_imgs = [Image.fromarray(np.uint8(np.clip((self.question_img + p) * 255, 0, 255))) for p in P]
        print("Pil imgs", pil_imgs)
        batch_paths, _ = self.retriever.flow_search(pil_imgs, self.question, 20, 100)

        fitness_scores = []
        for paths in batch_paths:
            scores = [1 / i if self.is_gt(paths[i - 1]) else 0 for i in range(1, 21)]
            fitness_scores.append(np.sum(scores))
        return np.array(fitness_scores)

    def is_gt(self, path):
        return self.gt_paths.get(path, False)

    def tournament_selection(self, fitness_scores, k=4):
        indices = np.random.choice(len(fitness_scores), k, replace=False)
        return indices[np.argmin(fitness_scores[indices])]

    def solve(self):
        w, h, c = self.np_question_img.shape
        P = np.random.uniform(-self.epsilon, self.epsilon, size=(self.pop_size, w, h, c))
        print("P shape: ", P.shape)
        P_fitness = self.fitness(P)
        print("P_fitness shape: ", P_fitness.shape)
        
        for _ in tqdm(range(self.max_iteration)):
            parent_indices = np.random.randint(0, self.pop_size, size=(self.pop_size, 2))
            O = (P[parent_indices[:, 0]] + P[parent_indices[:, 1]]) / 2
            print("Ofstring shape: ", O.shape)
            mutation_mask = np.random.rand(self.pop_size, 1, 1, 1) < self.mutation_rate
            mutation_values = np.random.uniform(-self.epsilon, self.epsilon, size=(self.pop_size, w, h, c))
            O = np.where(mutation_mask, mutation_values, O)
            O = np.clip(O, -self.epsilon, self.epsilon)

            O_fitness = self.fitness(O)
            print("P_fitness shape: ", O_fitness.shape)

            pool = np.concatenate((P, O), axis=0)
            pool_fitness = np.concatenate((P_fitness, O_fitness), axis=0)

            print("pool shape: ", pool.shape)
            print("pool_fitness shape: ", pool_fitness.shape)

            selected_P = []
            selected_fitness = []
            for _ in range(self.pop_size):
                idx = self.tournament_selection(pool_fitness)
                selected_P.append(pool[idx])
                selected_fitness.append(pool_fitness[idx])

            P = np.stack(selected_P)
            P_fitness = np.array(selected_fitness)

        best_idx = np.argmin(P_fitness)
        return P[best_idx], P_fitness[best_idx]

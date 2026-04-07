import os
import pickle
import numpy as np
import faiss


class FaissIndexManager:
    """
    Manage a FAISS index for nearest-neighbor search of experience vectors.
    """

    def __init__(self, dim: int, use_gpu: bool = False):
        self.dim = dim
        self.use_gpu = use_gpu

        self.index = faiss.IndexFlatL2(dim)

        if self.use_gpu:
            gpu_res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(gpu_res, 0, self.index)

        self.id_to_experience = {}
        self.current_count = 0

    def add(self, vectors: np.ndarray, experiences: list):
        vectors = np.asarray(vectors, dtype=np.float32)
        batch_size = vectors.shape[0]

        self.index.add(vectors)

        for i in range(batch_size):
            vec_id = self.current_count + i
            self.id_to_experience[vec_id] = experiences[i]

        self.current_count += batch_size

    def search(self, query_vectors: np.ndarray, top_k: int = 5):
        query_vectors = np.asarray(query_vectors, dtype=np.float32)
        distances, indices = self.index.search(query_vectors, top_k)

        results = []
        for idx_row in indices:
            row_exps = [self.id_to_experience.get(vec_id) for vec_id in idx_row]
            results.append(row_exps)

        return distances, results

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        else:
            cpu_index = self.index

        faiss.write_index(cpu_index, os.path.join(save_dir, "faiss_index.index"))

        with open(os.path.join(save_dir, "id_to_experience.pkl"), "wb") as f:
            pickle.dump(self.id_to_experience, f)

        print(f"✅ FAISS index and experience mapping saved to {save_dir}")

    def load(self, load_dir: str):
        index_cpu = faiss.read_index(os.path.join(load_dir, "faiss_index.index"))

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        else:
            self.index = index_cpu

        with open(os.path.join(load_dir, "id_to_experience.pkl"), "rb") as f:
            exp_dict = pickle.load(f)

        self.id_to_experience = exp_dict
        self.current_count = len(exp_dict)

        print(f"✅ Loaded FAISS index and experiences from {load_dir}")
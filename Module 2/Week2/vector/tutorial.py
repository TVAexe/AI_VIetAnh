import numpy as np
def vector_length(vector):
    return np.linalg.norm(vector)

def dot_product(vector1, vector2):
    return np.dot(vector1, vector2)

def matrix_multi_vector(matrix, vector):
    return np.dot(matrix, vector)

def matrix_multi_matrix(matrix1, matrix2):
    return np.dot(matrix1, matrix2)

def inverse_matrix(matrix):
    return np.linalg.inv(matrix)

def compute_eigenvalues_and_eigenvectors(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvalues, eigenvectors

def cosine_similar(vector1, vector2):
    dot_prod = dot_product(vector1, vector2)
    norm1 = vector_length(vector1)
    norm2 = vector_length(vector2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_prod / (norm1 * norm2)
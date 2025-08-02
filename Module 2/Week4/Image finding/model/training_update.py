import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from training import plot_results, read_image_from_path,folder_to_images,absolute_difference,mean_square_difference,cosine_similarity,correlation_coeficient
ROOT='../data'
CLASS_NAME=sorted(os.listdir(os.path.join(ROOT, 'train')))


embedding_function= OpenCLIPEmbeddingFunction()
def get_single_image_embeeding(image):
    embedding = embedding_function._encode_image(image)
    return np.array(embedding)


def get_new_l1_score(root_img_path, query_img_path, size):
    query = read_image_from_path(query_img_path, size)
    query_embedding = get_single_image_embeeding(query)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path=root_img_path + folder
            imgs_np, imgs_path = folder_to_images(path, size)
            embedding_list=[]
            for idx_img in range(imgs_np.shape[0]):
                img_embedding = get_single_image_embeeding(imgs_np[idx_img].astype(np.uint8))
                embedding_list.append(img_embedding)

            rates = absolute_difference(query_embedding, np.stack(embedding_list))
            ls_path_score.extend(list(zip(imgs_path, rates.tolist())))
    return query, ls_path_score

root_img_path = f"{ROOT}/train/"
query_img_path = f"{ROOT}/test/African_crocodile/n01697457_18534.JPEG"
size = (448,448)
query, ls_path_score = get_new_l1_score(root_img_path, query_img_path, size)
plot_results(query_img_path, ls_path_score, reverse=False)

def get_new_l2_score(root_img_path, query_img_path, size):
    query = read_image_from_path(query_img_path, size)
    query_embedding = get_single_image_embeeding(query)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            imgs_np, imgs_path = folder_to_images(path, size)
            embedding_list = []
            for idx_img in range(imgs_np.shape[0]):
                img_embedding = get_single_image_embeeding(imgs_np[idx_img].astype(np.uint8))
                embedding_list.append(img_embedding)

            rates = mean_square_difference(query_embedding, np.stack(embedding_list))
            ls_path_score.extend(list(zip(imgs_path, rates.tolist())))
    return query, ls_path_score

root_img_path = f"{ROOT}/train/"
query_img_path = f"{ROOT}/test/African_crocodile/n01697457_18534.JPEG"
size = (448,448)
query, ls_path_score = get_new_l2_score(root_img_path, query_img_path, size)
plot_results(query_img_path, ls_path_score, reverse=False)

def get_new_cosine_score(root_img_path, query_img_path, size):
    query = read_image_from_path(query_img_path, size)
    query_embedding = get_single_image_embeeding(query)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            imgs_np, imgs_path = folder_to_images(path, size)
            embedding_list = []
            for idx_img in range(imgs_np.shape[0]):
                img_embedding = get_single_image_embeeding(imgs_np[idx_img].astype(np.uint8))
                embedding_list.append(img_embedding)

            rates = cosine_similarity(query_embedding, np.stack(embedding_list))
            ls_path_score.extend(list(zip(imgs_path, rates.tolist())))
    return query, ls_path_score

root_img_path = f"{ROOT}/train/"
query_img_path = f"{ROOT}/test/African_crocodile/n01697457_18534.JPEG"
size = (448,448)
query, ls_path_score = get_new_cosine_score(root_img_path, query_img_path, size)
plot_results(query_img_path, ls_path_score, reverse=True)

def get_new_correlation_score(root_img_path, query_img_path, size):
    query = read_image_from_path(query_img_path, size)
    query_embedding = get_single_image_embeeding(query)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            imgs_np, imgs_path = folder_to_images(path, size)
            embedding_list = []
            for idx_img in range(imgs_np.shape[0]):
                img_embedding = get_single_image_embeeding(imgs_np[idx_img].astype(np.uint8))
                embedding_list.append(img_embedding)

            rates = correlation_coeficient(query_embedding, np.stack(embedding_list))
            ls_path_score.extend(list(zip(imgs_path, rates.tolist())))
    return query, ls_path_score

root_img_path = f"{ROOT}/train/"
query_img_path = f"{ROOT}/test/African_crocodile/n01697457_18534.JPEG"
size = (448,448)
query, ls_path_score = get_new_correlation_score(root_img_path, query_img_path, size)
plot_results(query_img_path, ls_path_score, reverse=True)

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

ROOT='../data'
CLASS_NAME=sorted(os.listdir(os.path.join(ROOT, 'train')))

def read_image_from_path(path,size):
    imgs=Image.open(path).convert('RGB').resize(size)
    return np.array(imgs)

def folder_to_images(folder, size):
    list_dir= [folder+'/'+name for name in os.listdir(folder)]
    imgs_np=np.zeros(shape=(len(list_dir), *size, 3))
    imgs_path=[]
    for i, path in enumerate(list_dir):
        imgs_np[i] = read_image_from_path(path, size)
        imgs_path.append(path)
    imgs_path=np.array(imgs_path)
    return imgs_np, imgs_path

def plot_results(query_img_path, ls_path_score, reverse):
    fig = plt.figure(figsize=(15, 9))
    fig.add_subplot(2, 3, 1)
    plt.imshow(read_image_from_path(query_img_path, size=(448,448)))
    plt.title(f"Query Image: {query_img_path.split('/')[-2]}", fontsize=16)
    plt.axis("off")
    for i, (img_path, score) in enumerate(sorted(ls_path_score, key=lambda x : x[1], reverse=reverse)[:5], 2):
        fig.add_subplot(2, 3, i)
        plt.imshow(read_image_from_path(img_path, size=(448,448)))
        plt.title(f"Top {i-1}: {img_path.split('/')[-2]}", fontsize=16)
        plt.axis("off")
    plt.show()
    

def absolute_difference(query,data):
    axis_batch_size=tuple(range(1, len(query.shape)))
    return np.sum(np.abs(query - data), axis=axis_batch_size)

def get_l1_score(root_img_path, query_img_path, size):
    query= read_image_from_path(query_img_path, size)
    ls_path_score=[]
    for folder in CLASS_NAME:
        path=root_img_path+ folder
        imgs_np, imgs_path = folder_to_images(path, size)
        rates= absolute_difference(query, imgs_np)
        ls_path_score.extend(list(zip(imgs_path, rates.tolist())))
    return query, ls_path_score

    


def mean_square_difference(query, data):
    axis_batch_size=tuple(range(1, len(query.shape)))
    return np.mean((query - data)**2, axis=axis_batch_size)

def get_l2_score(root_img_path, query_img_path, size):
    query = read_image_from_path(query_img_path, size)
    ls_path_score = []
    for folder in CLASS_NAME:
        path = root_img_path + folder
        imgs_np, imgs_path = folder_to_images(path, size)
        rates = mean_square_difference(query, imgs_np)
        ls_path_score.extend(list(zip(imgs_path, rates.tolist())))
    return query, ls_path_score




def cosine_similarity(query, data):
    axis_batch_size = tuple(range(1, len(query.shape)))
    query_norm=np.sqrt(np.sum(query**2))
    data_norm=np.sqrt(np.sum(data**2, axis=axis_batch_size))
    return np.sum(query * data, axis=axis_batch_size) / (query_norm * data_norm+np.finfo(float).eps)

def get_cosine_score(root_img_path, query_img_path, size):
    query = read_image_from_path(query_img_path, size)
    ls_path_score = []
    for folder in CLASS_NAME:
        path = root_img_path + folder
        imgs_np, imgs_path = folder_to_images(path, size)
        rates = cosine_similarity(query, imgs_np)
        ls_path_score.extend(list(zip(imgs_path, rates.tolist())))
    return query, ls_path_score



def correlation_coeficient(query, data):
    axis_batch_size = tuple(range(1, len(query.shape)))
    query_mean = query-np.mean(query)
    data_mean = data-np.mean(data, axis=axis_batch_size, keepdims=True)
    query_norm= np.sqrt(np.sum(query_mean**2))
    data_norm = np.sqrt(np.sum(data_mean**2, axis=axis_batch_size))
    return np.sum(query_mean * data_mean, axis=axis_batch_size) / (query_norm * data_norm + np.finfo(float).eps)

def get_correlation_score(root_img_path, query_img_path, size):
    query = read_image_from_path(query_img_path, size)
    ls_path_score = []
    for folder in CLASS_NAME:
        path = root_img_path + folder
        imgs_np, imgs_path = folder_to_images(path, size)
        rates = correlation_coeficient(query, imgs_np)
        ls_path_score.extend(list(zip(imgs_path, rates.tolist())))
    return query, ls_path_score


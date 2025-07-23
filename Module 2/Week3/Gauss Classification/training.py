import numpy as np
from sympy import comp

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = []
        for line in file:
            row = line.strip().split(',')
            data.append(row)
        return np.array(data)
    

def compute_prior_probablity(data):
    y_unique = np.unique(data[:, -1])
    prior_probability = np.zeros(len(y_unique))
    for i, label in enumerate(y_unique):
        prior_probability[i] = np.sum(data[:, -1] == label) / len(data)
    return prior_probability
    

def gaussian_probability(x, mean, std):
    exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    return (1 / (std * np.sqrt(2 * np.pi))) * exponent

def compute_mean_variance(data):
    y_unique = np.unique(data[:, -1])
    num_features = data.shape[1] - 1
    mean = np.zeros((len(y_unique), num_features))
    variance = np.zeros((len(y_unique), num_features))
    
    for i, label in enumerate(y_unique):
        subset = data[data[:, -1] == label][:, :-1].astype(float)
        mean[i] = np.mean(subset, axis=0)
        variance[i] = np.var(subset, axis=0)

    return mean, variance

def train_gaussian_naive_bayes(data):
    prior_probability = compute_prior_probablity(data)
    mean, variance = compute_mean_variance(data)
    return prior_probability, mean, variance

def predict_gaussian_naive_bayes(X, prior_probability, mean, variance,data):
    y_unique = np.unique(data[:, -1])
    num_features = len(X)
    probabilities = np.zeros((len(y_unique), num_features))
    
    for i, label in enumerate(y_unique):
        for j in range(num_features):
            probabilities[i, j] = gaussian_probability(X[j], mean[i, j], np.sqrt(variance[i, j]))
    
    class_probabilities = prior_probability * np.prod(probabilities, axis=1)
    predicted_class_index = np.argmax(class_probabilities)
    
    return y_unique[predicted_class_index]

data = load_data('iris.data.txt')
prior_probability, mean, variance = train_gaussian_naive_bayes(data)
X = [6.3 , 3.3 , 6.0 , 2.5]
predicted_class = predict_gaussian_naive_bayes(X, prior_probability, mean, variance,data)
print(f'Predicted class for {X} is: {predicted_class}')

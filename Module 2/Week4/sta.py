import numpy as np
import pandas as pd

def compute_mean(X):
    X= np.array(X)
    return np.mean(X)

def compute_median(X):
    X= np.array(X)
    return np.median(X)

# độ lệch chuẩn
def compute_std(X):
    X= np.array(X)
    return np.std(X)

def compute_variance(X):
    X= np.array(X)
    return np.var(X)

# tính hệ số tương quan
def compute_correlation_coefficient(X, Y):
    X = np.array(X)
    Y = np.array(Y)
    return np.corrcoef(X, Y)[0, 1]


data = pd.read_csv("advertising.csv")
X = data['TV']
Y = data['Radio']
print(data.corr())
print(np.corrcoef(X, Y))
theta = compute_correlation_coefficient(X, Y)
print(f"Hệ số tương quan giữa TV và Radio: {theta}")


features = ['TV', 'Radio', 'Newspaper']
for feature1 in features:
    for feature2 in features:
        corr = compute_correlation_coefficient(data[feature1], data[feature2])
        print(f"Hệ số tương quan giữa {feature1} và {feature2}: {corr}")

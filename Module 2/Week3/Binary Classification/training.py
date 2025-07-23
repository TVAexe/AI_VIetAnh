import numpy as np

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = []
        for line in file:
            row = line.strip().split(',')
            if len(row) > 1:  
                data.append(row[1:]) 
        return np.array(data)


def compute_prior_probablity(data):
    y_unique = ['No', 'Yes']
    prior_probability = np.zeros (len( y_unique ))
    for i, label in enumerate(y_unique):
        prior_probability[i] = np.sum(data[:, -1] == label) / len(data)
    return prior_probability





def compute_conditional_probability(train_data):
    y_unique = np.array(['No', 'Yes'])
    num_features = train_data.shape[1] - 1
    
    conditional_probability = np.empty(num_features, dtype=object)
    list_x_name = np.empty(num_features, dtype=object)
    
    for i in range(num_features):
        x_unique = np.unique(train_data[:, i])
        list_x_name[i] = x_unique
        
        # Tạo ma trận xác suất cho feature i
        prob_matrix = np.zeros((len(y_unique), len(x_unique)))
        
        for j, label in enumerate(y_unique):
            for k, x_value in enumerate(x_unique):
                count = np.sum((train_data[:, i] == x_value) & (train_data[:, -1] == label))
                prob_matrix[j, k] = count / np.sum(train_data[:, -1] == label)
        
        conditional_probability[i] = prob_matrix
    
    return conditional_probability, list_x_name


def get_index_from_value (feature_name,list_features ) :
    return np.where (list_features == feature_name )[0][0]


def train_naive_bayes (train_data):
    y_unique = ['No', 'Yes']
    prior_probability = compute_prior_probablity(train_data)
    conditional_probability, list_x_name = compute_conditional_probability(train_data)
    return prior_probability, conditional_probability, list_x_name

def prediction_play_tennis (X , list_x_name , prior_probability , conditional_probability ):
    x1 = get_index_from_value ( X [0] , list_x_name [0])
    x2 = get_index_from_value ( X [1] , list_x_name [1])
    x3 = get_index_from_value ( X [2] , list_x_name [2])
    x4 = get_index_from_value ( X [3] , list_x_name [3])
    
    p_no = prior_probability[0] * conditional_probability[0][0, x1] * conditional_probability[1][0, x2] * conditional_probability[2][0, x3] * conditional_probability[3][0, x4]
    p_yes = prior_probability[1] * conditional_probability[0][1, x1] * conditional_probability[1][1, x2] * conditional_probability[2][1, x3] * conditional_probability[3][1, x4]

    print(p_no, p_yes)
    if p_no > p_yes:
        return 'No'
    else:
        return 'Yes'


data = load_data('data.txt')
prior_probability, conditional_probability, list_x_name = train_naive_bayes(data)
X = ['Sunny', 'Cool', 'High', 'Strong']
result = prediction_play_tennis(X, list_x_name, prior_probability, conditional_probability)
print(result)

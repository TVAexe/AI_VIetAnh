import numpy as np

def least_square_estimation(vector_x, vector_y):
    A=np.vstack((vector_x, np.ones(len(vector_x)))).T

    w,b= np.linalg.lstsq(A, vector_y, rcond=None)[0]
    return w, b

x= np.array([6.7, 4.6, 3.5, 5.5])
y= np.array([9.1, 5.9, 4.6, 6.7])

w, b = least_square_estimation(x, y)
print(f"Estimated slope (w): {w}")
print(f"Estimated intercept (b): {b}")


def demo():
    # Ax=Y
    # AT A x = AT Y
    # x= (AT A)^-1 AT Y

    A=np.array([[6.7,1],[4.6,1],[3.5,1],[5.5,1]])
    B=np.array([9.1,5.9,4.6,6.7])
    
    ATA=np.dot(A.T, A)
    ATA_1=np.linalg.inv(ATA)
    x=np.dot(ATA_1, np.dot(A.T, B))
    print(x)

demo()
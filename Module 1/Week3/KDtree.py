import matplotlib.pyplot as plt
import math
class BinaryTree():
    def __init__(self, vector):
        self.vector = vector
        self.left = None
        self.right = None

def distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def build_kd_tree(dataList, depth=0):
    if not dataList:
        return None

    k = len(dataList[0])  # Number of dimensions
    axis = depth % k  # Current axis to split on

    # Sort the data list and choose the median as the pivot element
    dataList.sort(key=lambda x: x[axis])
    median_index = len(dataList) // 2

    # Create a node for the median
    node = BinaryTree(dataList[median_index])

    # Recursively build the left and right subtrees
    node.left = build_kd_tree(dataList[:median_index], depth + 1)
    node.right = build_kd_tree(dataList[median_index + 1:], depth + 1)

    return node
    
def nearest_neighbor(node, point, depth=0, best=None):
    if node is None:
        return best

    k = len(point)
    axis = depth % k

    here_dist = distance(point, node.vector)
    if best is None or here_dist < best[0]:
        best = (here_dist, node.vector)

    # Quyết định nhánh đi trước
    if point[axis] < node.vector[axis]:
        best = nearest_neighbor(node.left, point, depth + 1, best)
        # Kiểm tra nhánh còn lại nếu cần
        if abs(point[axis] - node.vector[axis]) < best[0]:
            best = nearest_neighbor(node.right, point, depth + 1, best)
    else:
        best = nearest_neighbor(node.right, point, depth + 1, best)
        if abs(point[axis] - node.vector[axis]) < best[0]:
            best = nearest_neighbor(node.left, point, depth + 1, best)
    return best
# Vẽ các điểm

def plot_kd_tree(node, xmin, xmax, ymin, ymax, depth=0):
    if node is None:
        return
    axis = depth % 2
    x, y = node.vector

    if axis == 0:
        # Chia theo x, vẽ đường dọc
        plt.plot([x, x], [ymin, ymax], 'r--')
        plot_kd_tree(node.left, xmin, x, ymin, ymax, depth + 1)
        plot_kd_tree(node.right, x, xmax, ymin, ymax, depth + 1)
    else:
        # Chia theo y, vẽ đường ngang
        plt.plot([xmin, xmax], [y, y], 'b--')
        plot_kd_tree(node.left, xmin, xmax, ymin, y, depth + 1)
        plot_kd_tree(node.right, xmin, xmax, y, ymax, depth + 1)



dataList = [[1,2] , [2, 6], [3, 4], [5, 6], [8, 3], [7, 8]]
kd_tree = build_kd_tree(dataList)
xs, ys = zip(*dataList)
plt.scatter(xs, ys, c='black')

# Vẽ KD-tree splits
plot_kd_tree(kd_tree, xmin=min(xs)-1, xmax=max(xs)+1, ymin=min(ys)-1, ymax=max(ys)+1)

plt.xlabel('x')
plt.ylabel('y')
plt.title('2D KD-Tree Visualization')
plt.grid(True)
plt.show()


query_point = [4, 5]
dist, nearest = nearest_neighbor(kd_tree, query_point)
print(f"Nearest point to {query_point} is {nearest} with distance {dist:.2f}")
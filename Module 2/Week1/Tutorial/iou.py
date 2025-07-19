def iou_calculation(box_1, box_2):
    x_A=max(box_1[0], box_2[0])
    y_A=max(box_1[1], box_2[1])
    x_B=min(box_1[2], box_2[2])
    y_B=min(box_1[3], box_2[3])

    intersection_area=max(0, x_B-x_A+1)*max(0, y_B-y_A+1)
    box_1_area=(box_1[2]-box_1[0]+1)*(box_1[3]-box_1[1])
    box_2_area=(box_2[2]-box_2[0])*(box_2[3]-box_2[1])
    iou=intersection_area/(box_1_area+box_2_area-intersection_area)
    return iou

import numpy as np
def iou_calculation_numpy(box_1, box_2):
    box_1 = np.array(box_1)
    box_2 = np.array(box_2)

    x_A = np.maximum(box_1[0], box_2[0])
    y_A = np.maximum(box_1[1], box_2[1])
    x_B = np.minimum(box_1[2], box_2[2])
    y_B = np.minimum(box_1[3], box_2[3])

    intersection_area = np.maximum(0, x_B - x_A + 1) * np.maximum(0, y_B - y_A + 1)
    box_1_area = (box_1[2] - box_1[0] + 1) * (box_1[3] - box_1[1] + 1)
    box_2_area = (box_2[2] - box_2[0] + 1) * (box_2[3] - box_2[1] + 1)

    iou = intersection_area / (box_1_area + box_2_area - intersection_area)
    return iou

def iou_for_all(boxed_final, boxes):
    boxes= np.array(boxes)
    x1= boxes[:, 0]
    y1= boxes[:, 1]
    x2= boxes[:, 2]
    y2= boxes[:, 3]
    x_A = np.maximum(boxed_final[0], x1)
    y_A = np.maximum(boxed_final[1], y1)
    x_B = np.minimum(boxed_final[2], x2)
    y_B = np.minimum(boxed_final[3], y2)

    intersection_area = np.maximum(0, x_B - x_A + 1) * np.maximum(0, y_B - y_A + 1)
    box_1_area = (boxed_final[2] - boxed_final[0] + 1) * (boxed_final[3] - boxed_final[1] + 1)
    box_2_area = (x2 - x1 + 1) * (y2 - y1 + 1)

    iou = intersection_area / (box_1_area + box_2_area - intersection_area)
    return iou

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)

        if len(indices) == 1:
            break

        rest_indices = indices[1:]
        ious = iou_for_all(boxes[current], [boxes[i] for i in rest_indices])
        indices = [rest_indices[i] for i in range(len(ious)) if ious[i] <= iou_threshold]

    return keep
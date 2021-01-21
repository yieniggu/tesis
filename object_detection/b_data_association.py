"""
As implemented in https://github.com/abewley/sort but with some modifications

For each detected item, it computes the intersection over union (IOU) w.r.t. each tracked object. (IOU matrix)
Then, it applies the Hungarian algorithm (via linear_assignment) to assign each det. item to the best possible
tracked item (i.e. to the one with max. IOU).

Note: a more recent approach uses a Deep Association Metric instead.
see https://github.com/nwojke/deep_sort
"""

import numpy as np

def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def iou(bb_boxA,bb_boxB):

  """
  Computes IUO between two bboxes in the form [x1,y1,x2,y2]
  """
  
  xx1 = np.maximum(bb_boxA[0], bb_boxB[0])
  yy1 = np.maximum(bb_boxA[1], bb_boxB[1])
  xx2 = np.minimum(bb_boxA[2], bb_boxB[2])
  yy2 = np.minimum(bb_boxA[3], bb_boxB[3])
  
  width = np.maximum(0., xx2 - xx1)
  height = np.maximum(0., yy2 - yy1)
  wh = width * height
  o = wh / ((bb_boxA[2]-bb_boxA[0])*(bb_boxA[3]-bb_boxA[1])
    + (bb_boxB[2]-bb_boxB[0])*(bb_boxB[3]-bb_boxB[1]) - wh)
  return(o)

def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.1):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    # Si no hay trackers retornamos :
    # matched: [] , arreglo vacio de shape (0, 2)
    # unmatched_detections: [0, 1, 2, 3, 4] , arreglo con indices que indican las detecciones
    # unmatched_trackers: [], arreglo vacion de shape (0, 5)
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections))
  
    # Inicializamos la matriz de costo de IoU con ceros
    iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

    # Calculamos IoU entre los trackers y las detecciones
    for d,detection in enumerate(detections):
        for t,tracker in enumerate(trackers):
            print("[INFO] Calculating IoU between: \n", detection, " and: \t", tracker.get_state())
            iou_matrix[d,t] = iou(detection,tracker.get_state())
            print("[INFO] IoU: ", iou_matrix[d,t])
    '''The linear assignment module tries to minimise the total assignment cost.
    In our case we pass -iou_matrix as we want to maximise the total IOU between track predictions and the frame detection.'''
    print("[INFO] IoU Matrix: ", iou_matrix)
    
    matched_indices = linear_assignment(-iou_matrix)
    #print("[INFO] Matched Indices: ", matched_indices)

    #print("[INFO] Matched Indices again: ", matched_indices[:0])
    unmatched_detections = []
    for d,detection in enumerate(detections):
        # Si la deteccion no encuentra match la agregamos a las detecciones sin match
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    print("[INFO] Unmatched detections: ", unmatched_detections)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        print("[INFO] Checking treshold for detection: {} & tracker {} \n IoU: {}".format(m[0], m[1], iou_matrix[m[0], m[1]]))
        if(iou_matrix[m[0],m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections)
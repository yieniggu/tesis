"""
As implemented in https://github.com/abewley/sort but with some modifications
"""

import numpy as np
from filterpy.kalman import KalmanFilter


'''Motion Model'''
class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox,id, boundary):
    """
    Initialises a tracker using initial bounding box.

    F: State transition Matrix used in ẍ = Fx + Bu
    x: Filter state estimate
    u: Motion vector
    z: Measurement

    H: Measurement function
    B: Control transition matrix
    P: Uncertainty covariance
    Q: Process noise covariance
    R: Measurement noise covariance
    """
    #define constant velocity model
    self.kalman_filter = KalmanFilter(dim_x=7, dim_z=4)
    self.kalman_filter.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kalman_filter.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kalman_filter.R[2:,2:] *= 10.
    self.kalman_filter.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kalman_filter.P *= 10.
    self.kalman_filter.Q[-1,-1] *= 0.01
    self.kalman_filter.Q[4:,4:] *= 0.01

    # Inicializamos el state vector a partir de la primera medicion
    self.kalman_filter.x[:4] = convert_bbox_to_z(bbox)
    
    # Calculate centroid of bounding box
    centroid_x = int((bbox[0] + bbox[2]) / 2)
    centroid_y = int((bbox[1] + bbox[3]) / 2)
    self.first_centroid = np.array([centroid_x, centroid_y], dtype=int)
    self.last_centroid = np.array([centroid_x, centroid_y], dtype=int)
    self.id = id
    self.time_since_update = 0

    # Check if tracked object is initialized out of ROI
    if centroid_x > boundary:
      self.has_entered_ROI = True
      self.active = True
      self.initialized_in_ROI = True
    else:
      self.has_entered_ROI = False
      self.active = False
      self.initialized_in_ROI = False

    self.checked_entrance = False

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    if bbox != []:
        self.time_since_update = 0
        print("[KALMAN-INFO] Before updating on tracker {}: {}".format(self.id, self.get_state())) 
        self.kalman_filter.update(convert_bbox_to_z(bbox))
        print("[KALMAN-INFO] After updating on tracker {}: {}".format(self.id, self.get_state()))
        print("[KALMAN-INFO] New time since update: {}".format(self.time_since_update))

        # Obtenemos el nuevo estado
        bbox = self.get_state()
        
        # Calculate centroid of bounding box
        centroid_x = int((bbox[0] + bbox[2]) / 2)
        centroid_y = int((bbox[1] + bbox[3]) / 2)

        self.last_centroid = np.array([centroid_x, centroid_y], dtype=int)

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kalman_filter.x[6]+self.kalman_filter.x[2])<=0):
      self.kalman_filter.x[6] *= 0.0
    
    # Predecimos
    print("[KALMAN-INFO] Before prediction on tracker {}: {}".format(self.id, self.get_state()))
    self.kalman_filter.predict()
    print("[KALMAN-INFO] After prediction: on tracker {}: {}".format(self.id, self.get_state()))

    # Obtenemos el nuevo estado
    bbox = self.get_state().astype(int)
    
    # Calculate centroid of bounding box
    centroid_x = int((bbox[0] + bbox[2]) / 2)
    centroid_y = int((bbox[1] + bbox[3]) / 2)

    self.time_since_update += 1
    print("[KALMAN-INFO] New time since last update: {}".format(self.time_since_update))
    self.last_centroid = np.array([centroid_x, centroid_y], dtype=int)

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kalman_filter.x)[0].astype(int)


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  # Asignamos cada variable del vector a una variable descriptiva
  xmin = bbox[0]
  ymin = bbox[1]
  xmax = bbox[2]
  ymax = bbox[3]

  # Obtenemos los valores para el vector de estado
  width = xmax-xmin
  height = ymax-ymin
  centroid_x_location = xmin+width/2.
  centroid_y_location = ymin+height/2.
  scale = width*height    #scale is just area
  aspect_ratio = width/float(height)

  return np.array([centroid_x_location,centroid_y_location,scale,aspect_ratio]).reshape((4,1)) # Convert to column vector

def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  # Asignamos cada variable del vector a una variable descriptiva
  centroid_x_location = x[0]
  centroid_y_location = x[1]
  scale = x[2]
  aspect_ratio = x[3]

  # Obtenemos los valores para la bounding box
  width = np.sqrt(scale*aspect_ratio)
  height = scale/width
  if(score==None):
    return np.array([x[0]-width/2.,x[1]-height/2.,x[0]+width/2.,x[1]+height/2.]).reshape((1,4))
  else:
    return np.array([x[0]-width/2.,x[1]-height/2.,x[0]+width/2.,x[1]+height/2.,score]).reshape((1,5))
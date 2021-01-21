"""
As implemented in https://github.com/abewley/sort but with some modifications
"""

from __future__ import print_function
import numpy as np
from b_kalman_tracker import KalmanBoxTracker
from b_data_association import associate_detections_to_trackers
from screeninfo import get_monitors

class Sort:

	def __init__(self, dims, max_time=20):
		"""
		Sets key parameters for SORT
		"""
		self.trackers = []
		self.total_trackers = 0
		self.created_trackers = 0
		self.active_trackers = 0
		self.boundaries = self.set_boundaries(dims)
		self.max_time = max_time
		self.image_dims = dims

	def set_screen_size(self):
		monitor = get_monitors()[0]
		VIDEO_WIDTH = int(monitor.width - monitor.width * 0.035) # Take a litle space for the window frame
		VIDEO_HEIGHT = int(monitor.height - monitor.height * 0.06)

		screen_size = (VIDEO_WIDTH, VIDEO_HEIGHT)
		return screen_size

	def set_boundaries(self, dims, left_prop=0.3, right_prop=0.8):
		left_bound = int (dims[0] * left_prop)
		right_bound = int (dims[0] * right_prop)
		mid_bound = int (dims[0] * 0.5)

		self.boundaries = boundaries = np.array([left_bound, mid_bound, right_bound])
		return boundaries

	def update(self, detections):
		"""
		Params:
		detections - a numpy array of detections in the format [[x1, y1, x2,y2],[x1,y1,x2,y2],...]
		Requires: this method must be called once for each frame even with empty detections.
		Returns the a similar array, where the last column is the object ID.

		NOTE: The number of objects returned may differ from the number of detections provided.
		"""

		# Predecimos para cada tracker y actualizamos su informacion
		
		for t, _ in enumerate(self.trackers):
			#print("[INFO] Before predicting tracker: ", self.trackers[t].get_state())
			self.trackers[t].predict()
			#print("[INFO] After predicting tracker", self.trackers[t].get_state())

		# Asociamos las detecciones a algun tracker o creamos un nuevo tracker
		if detections.size != 0:
			print("[BSORT-INFO] Found new detections: ", detections)
			print("[BSORT-INFO] Attempting to associate or create tracker...")
			
			# Obtenemos las detecciones que pertenecen a un track y las que no
			matched, unmatched_detections = associate_detections_to_trackers(detections, self.trackers)
		
			#print("[INFO FROM BSORT] Matched: {}",format(matched))
			#print("[INFO FROM BSORT] UNMatched: {}",format(unmatched_detections))

			# Actualizamos el estado del tracker utilizando la deteccion asociada
			for match in matched:
				print("[BSORT-INFO] Updating tracker [{}]: {} with asociated detection ({})".format(self.trackers[match[1]].id, self.trackers[match[1]].get_state(), detections[match[0]]))
				self.trackers[match[1]].update(detections[match[0]])

			# Creamos un nuevo tracker basado en una nueva deteccion
			for unmatch in unmatched_detections:
				# Calculate Centroid of object
				centroid_x = (detections[unmatch][0] + detections[unmatch][2])/2 
				# Verificamos que no sea un objeto que se esta devolviendo
				if centroid_x > self.boundaries[0]:
					print("[BSORT-INFO] Tracker initialized inside ROI")
					self.total_trackers += 1
				else:
					print("[BSORT-INFO] Tracker initialized outside ROI")

					print("[BSORT-INFO] Creating new tracker on location {}".format(detections[unmatch]))
				self.created_trackers += 1
				new_tracker = KalmanBoxTracker(detections[unmatch], self.created_trackers, self.boundaries[0])
				print("[BSORT-INFO] New tracker {} created with id {}".format(new_tracker.get_state(), new_tracker.id))
				self.trackers.append(new_tracker)      

		else:
			print("[BSORT-INFO] No detections found")

		return self.trackers


	def update_trackers_state(self):
		"""
		Updates trackers state based on boundary
		"""
		self.active_trackers = 0

		for t,tracker in enumerate(self.trackers):
			if tracker.last_centroid[0] < self.boundaries[0]:
				tracker.active = False # In case object left detection zone
			else:
				tracker.has_entered_ROI = True
				tracker.active = True # In case object return detection zone
				self.active_trackers += 1

			# Check if tracker get lost on time
			#print("[INFO] tracker.time_since_update {} > {} self.max_time".format(tracker.time_since_update, self.max_time))
			if tracker.time_since_update > self.max_time:
				# Tracker initialized in or out ROI and lost track [Cases 1, 3]
				if tracker.last_centroid[0] < self.boundaries[0]:
					print("\n[BSORT-LOST-INFO] Tracker {} has left FOV area or lost outsite ROI - Removing but not discounting\n".format(tracker.id))
					del(self.trackers[t])
							
				# Tracker initialized out of ROI (possible mismatch) and los track inside left side ROI
				elif(not tracker.initialized_in_ROI) and (tracker.last_centroid[0] < self.boundaries[1]) and (tracker.last_centroid[0] > self.boundaries[0]):
					print("\n[BSORT-LOST-INFO] Tracker {} missmatched out of ROI has lost - Removing but not discounting\n".format(tracker.id))
					del(self.trackers[t])

				# Tracker initialized in or out of ROI and lost track
				else:
					print("\n[BSORT-LOST-INFO] Tracker {} has lost - Removing and discounting\n".format(tracker.id))
					del(self.trackers[t])
					self.total_trackers -= 1
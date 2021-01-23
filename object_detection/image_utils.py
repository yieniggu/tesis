import cv2
from PIL import Image
import numpy as np
import time
from random import randint
import colorsys
import random
import tensorflow as tf
from functools import partial
from screeninfo import get_monitors
import sys
import os

def read_image_from_pil(image_path):
	image = Image.open(image_path)

	print("[IMAGEUTILS] Succesfuly loaded image into PIL image from {}".format(image_path))
	print("[IMAGEUTILS] Image type {}".format(type(image)))

	return image

def read_image_from_cv2(image_path):
	image = cv2.imread(image_path)

	print("[IMAGEUTILS] Succesfuly loaded image into cv2 image from {}".format(image_path))
	print("[IMAGEUTILS] Image type {}".format(type(image)))

	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
	return image

def image_to_np(image):
	image_as_nparray = np.array(image)
	print("[IMAGEUTILS] Succesfully converted image no numpy array")
	print("[IMAGEUTILS] Image type {} | Image dims {}".format(type(image_as_nparray), image_as_nparray.shape))

	return image_as_nparray

def load_image_as_np(image_path, src="PIL"):
	if src == "PIL":
		start_pil = time.time()
		image_as_np = image_to_np(read_image_from_pil(image_path))
		print("[IMAGEUTILS] PIL image loading took {} ms".format(((time.time()-start_pil)*1000)))

	else:
		start_cv2 = time.time()
		image_as_np = image_to_np(read_image_from_cv2(image_path))
		total_cv2 = (time.time()-start_cv2)*1000
		print("[IMAGEUTILS] cv2 image loading took {} ms".format(total_cv2))

	return image_as_np, total_cv2

def preprocess_fn(image_path, input_size):
	# read image with tf
	image, _ = get_image_tf(image_path)
	if input_size is not None:
		image = resize_image_tf(image, input_size)
	
	return image

def get_dataset_tf(image_path=None, input_size=None, tensor_type='int'):
	print("[IMGUTILS] Starting tf dataset pipeline")
	print("[IMGUTILS] Creating dataset object from tensor_slices")
	dataset_load_start = time.time()

	# Load data with tf dataset pipeline
	if image_path is not None:
		dataset = tf.data.Dataset.from_tensor_slices([image_path])
		dataset_load_total = (time.time()-dataset_load_start)*1000
		print("[IMGUTILS] Done!, it took {}ms".format(dataset_load_total))
		
		# map the preprocess function
		def tf_preprocess_fn(image_path):
			image = tf.io.read_file(image_path)
			image = tf.image.decode_jpeg(image, channels=3)

			if input_size is not None:
				print("[IMGUTILS] Input size from get dataset tf {} - type {}"
						.format(input_size, type(input_size)))
				image = tf.image.resize(image, size=(input_size, input_size))
			if tensor_type == 'int':
				image = tf.cast(image, tf.uint8)
			else:
				image = tf.cast(image, tf.float32)
			return image

		print("[IMGUTILS] Starting preprocessing...")
		preprocessing_start = time.time()
		dataset = dataset.map(map_func=tf_preprocess_fn, num_parallel_calls=8)
		dataset = dataset.batch(1)
		dataset = dataset.repeat(count=1)
		preprocessing_total = (time.time()-preprocessing_start)*1000
		print("[IMGUTILS] Preprocessing done!, it took {}ms".format(preprocessing_total))

	else: 	
		print("[DATASET] Using syntethic data...")
		print("[DATASET] Creating features...")

		features = np.random.normal(loc=112, scale=70,
				size=(1, input_size, input_size, 3)).astype(np.float32)

		dataset_load_total = (time.time()-dataset_load_start)*1000
		print("[IMGUTILS] Done!, it took {}ms".format(dataset_load_total))

		if tensor_type == 'int':
			features = np.clip(features, 0.0, 255.0).astype(np.float32)
		else:
			features = np.clip(features, 0.0, 255.0).astype(np.float32)

		print("[IMGUTILS] Starting preprocessing...")
		preprocessing_start= time.time()
		features = tf.convert_to_tensor(value=tf.compat.v1.get_variable(
							"features", initializer=tf.constant(features)))

		print("[DATASET] Creating dataset from features...")
		dataset = tf.data.Dataset.from_tensor_slices([features])
		dataset = dataset.repeat(count=1)
		preprocessing_total = (time.time()-preprocessing_start)*1000
		print("[IMGUTILS] Preprocessing done!, it took {}ms".format(preprocessing_total))


	return dataset, dataset_load_total, preprocessing_total

def get_image_tf(image_path):
	print("[IMGUTILS] Starting tf image reading using tf")
	print("[IMGUTILS] Loading image from path...")
	loading_start = time.time()
	image = tf.io.read_file(image_path)
	loading_total = (time.time()-loading_start)*1000
	print("[IMGUTILS] Image loaded from path, it took {}ms".format(loading_total))

	print("[IMGUTILS] Decoding image...")
	decoding_start = time.time()
	image = tf.image.decode_jpeg(image, channels=3)
	decoding_total = (time.time()-decoding_start)*1000
	print("[IMGUTILS] Done!, it took {}ms".format(decoding_total))

	total_time = loading_total+decoding_total
	return image, total_time

def resize_image_tf(image, input_size):
	# resize image
	resized_image = tf.image.resize(image, size=(input_size, input_size))
	# cast image to uint8
	resized_image = tf.cast(image, tf.uint8)

	return resized_image

def resize_image(image, dims):
	start_resizing = time.time()
	height, width, _ = image.shape
	print("[IMAGEUTILS] Resizing image with dims {}x{} to {}x{}"
			.format(height, width, dims[0], dims[1]))
	image_resized = cv2.resize(image, (dims[0], dims[1]))
	print("[IMAGEUTILS] Image Resized succesfully, it took {} ms"
			.format((time.time()-start_resizing)*1000))

	return image_resized

def preprocess_image(image, target_size, mean=1.):

	ih, iw    = target_size
	h,  w, _  = image.shape

	scale = min(iw/w, ih/h)
	nw, nh  = int(scale * w), int(scale * h)
	image_resized = cv2.resize(image, (nw, nh))

	image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
	dw, dh = (iw - nw) // 2, (ih-nh) // 2
	image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized

	image_paded = image_paded / mean

	return image_paded

def draw_trackers_bounding_box(frame, label, object_tracked):
	draw_start = time.time()
	height, width, _ = frame.shape

	print("[IMAGEUTILS] Image dimensions are {}x{}".format(width, height))

	# set drawing color based on tracker status
	if object_tracked.active:
		bbox_color = (0, 128, 255)
	else:
		bbox_color = (10, 255, 0)

	# get tracked object bbox
	bbox = object_tracked.get_state().astype(int)

	# get tracker id to display in drawing
	tracker_id = object_tracked.id
	
	# Get bounding box boxdinates and draw box Interpreter can return
	#  boxdinates that are outside of image dimensions, 
	# need to force them to be within image using max() and min()
	ymin = int(bbox[1])  
	xmin = int(bbox[0])
	ymax = int(bbox[3])
	xmax = int(bbox[2])

	print("[IMAGEUTILS] Bbox found with coordinates: "
		"x1: {} - y1: {} - x2: {} - y2: {}".format(xmin, ymin, xmax, ymax))

	# set some display parameters
	bbox_thick = int(0.6 * (height + width) / 1000)
	if bbox_thick < 1: bbox_thick = 1
	fontScale = 0.75 * bbox_thick

	# prepare label text for display
	label_text = '[{}]: {}'.format(tracker_id, label)

	# Get font size
	(text_width, text_height), baseline = cv2.getTextSize(label_text, 
									cv2.FONT_HERSHEY_SIMPLEX, 
									fontScale, 2) 
	# put filled text rectangle
	cv2.rectangle(frame, (xmin, ymin), 
					(xmin + text_width, 
					ymin - text_height - baseline), 
					bbox_color, thickness=cv2.FILLED)
	# put text above rectangle
	cv2.putText(frame, label_text, 
					(xmin, ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
					fontScale, (255, 255, 0), bbox_thick, 
					lineType=cv2.LINE_AA)

	# Draw bounding box
	cv2.rectangle(frame, 
			(xmin, ymin), 
			(xmax, ymax), 
			(bbox_color), bbox_thick*2)

	# Calculate centroid of bounding box
	centroid = (object_tracked.last_centroid[0], object_tracked.last_centroid[1])    
	
	# Draw the centroid
	cv2.circle(frame, centroid, 6, (0, 0, 204), -1)

	print("[IMAGEUTILS] Draw finished. Drawing took {} ms"
			.format((time.time()-draw_start)*1000))

	return frame

def draw_bounding_box(image, bbox, label, score):
	draw_start = time.time()

	# get shape of image
	height, width, _ = image.shape
	print("[IMAGEUTILS] Image dimensions are {}x{}".format(width, height))

	# Get bounding box boxdinates and draw box Interpreter can return
	#  boxdinates that are outside of image dimensions, 
	# need to force them to be within image using max() and min()
	ymin = int(bbox[0])  
	xmin = int(bbox[1])
	ymax = int(bbox[2])
	xmax = int(bbox[3])

	print("[IMAGEUTILS] Bbox found with coordinates: "
		"x1: {} - y1: {} - x2: {} - y2: {}".format(xmin, ymin, xmax, ymax))

	# set some display parameters
	bbox_thick = int(0.6 * (height + width) / 1000)
	if bbox_thick < 1: bbox_thick = 1
	fontScale = 0.75 * bbox_thick
	bbox_color = (randint(0, 255), randint(0, 255), randint(0, 255))

	# prepare label text for display
	label_text = '{}: {}%'.format(label, int(score*100))

	# Get font size
	(text_width, text_height), baseline = cv2.getTextSize(label_text, 
									cv2.FONT_HERSHEY_SIMPLEX, 
									fontScale, 2) 
	# Make sure not to draw label too close to top of window
	"""label_ymin = max(ymin, text_width + 10) 
			cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), 
			(xmin+labelSize[0], label_ymin+baseLine-10), 
			bbox_color, cv2.FILLED) # Draw white box to put label text in
	"""
	# put filled text rectangle
	cv2.rectangle(image, (xmin, ymin), 
					(xmin + text_width, 
					ymin - text_height - baseline), 
					bbox_color, thickness=cv2.FILLED)
	# put text above rectangle
	cv2.putText(image, label_text, 
					(xmin, ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
					fontScale, (255, 255, 0), bbox_thick, 
					lineType=cv2.LINE_AA)

	# Draw bounding box
	cv2.rectangle(image, 
			(xmin, ymin), 
			(xmax, ymax), 
			(bbox_color), bbox_thick*2)

	print("[IMAGEUTILS] Draw finished. Drawing took {} ms"
			.format((time.time()-draw_start)*1000))

	return image


def resize_bounding_box(original_shape, scaled_shape, bbox):
	# get size of original and scaled input
	original_height, original_width, _ = original_shape
	scaled_height, scaled_width, _ = scaled_shape

	# get scale to transform bbox
	height_scale = scaled_height / original_height
	width_scale = scaled_width / original_width

	print("[IMAGEUTILS] Original bbox comes in "
		"x1: {} - y1: {} - x2: {} - y2: {}".format(bbox[1], bbox[0], bbox[3], bbox[2]))

	# resize bbox
	xmin = int(np.round(bbox[1]*width_scale))
	ymin = int(np.round(bbox[0]*height_scale))
	xmax = int(np.round(bbox[3]*width_scale))
	ymax = int(np.round(bbox[2]*height_scale))

	print("[IMAGEUTILS] Resized bbox results in "
		"x1: {} - y1: {} - x2: {} - y2: {}".format(xmin, ymin, xmax, ymax))

	return [ymin, xmin, ymax, xmax]

def draw_bounding_boxes(image, detections, labels, threshold=0.5, bbox_results=None, save_results=False):
	
	print('[IMAGEUTILS] Starting bbox drawing pipeline...')
	total_drawing_start = time.time()

	print("[IMAGEUTILS] Extracting output tensors metadata...")
	extraction_start = time.time()
	# since outputs are batches tensors and we're interested
	# in the first num_detections we remove the batch dimension
	num_detections = int(detections.pop('num_detections'))
	detections = {key: value[0, :num_detections].numpy()
				   for key, value in detections.items()}
	detections['num_detections'] = num_detections

	# detection_classes should be ints.
	detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
	
	# we copy the image to draw results on a new one
	image_np_with_detections = image.copy()

	# draw bbox, confidence and label
	boxes = detections['detection_boxes']
	classes = detections['detection_classes']
	scores = detections['detection_scores']

	print("[IMAGEUTILS] Metadata extracted succesfully")
	print("[IMAGEUTILS] Metadata extraction took {} ms".
					format((time.time()-extraction_start)*1000))
	#print("[IMAGEUTILS] boxes: ", boxes, type(boxes))
	#print("[IMAGEUTILS] classes: ", classes, type(classes))
	#print("[IMAGEUTILS] scores: ", scores, type(scores))


	print("[IMAGEUTILS] Starting to draw...")
	valid_detections = 0

	height, width, _ = image.shape

	# loop over all detections
	for i in range(len(scores)):
		# draw instances with minimum confidence only
		if ((scores[i] > threshold) and (scores[i] <=1)):
			valid_detections += 1
			# get box, and label
			box = boxes[i]

			# fix class labels
			label = labels[classes[i]-1]

			# Get bounding box boxdinates and draw box Interpreter can return
			#  boxdinates that are outside of image dimensions, 
			# need to force them to be within image using max() and min()
			ymin = int(max(1,(box[0] * height)))  
			xmin = int(max(1,(box[1] * width)))
			ymax = int(min(height,(box[2] * height)))
			xmax = int(min(width,(box[3] * width)))


			# set some display parameters
			bbox_thick = int(0.6 * (height + width) / 1000)
			if bbox_thick < 1: bbox_thick = 1
			fontScale = 0.75 * bbox_thick
			bbox_color = (randint(0, 255), randint(0, 255), randint(0, 255))

			# prepare label text for display
			label_text = '{}: {}%'.format(label, int(scores[i]*100))

			# Get font size
			(text_width, text_height), baseline = cv2.getTextSize(label_text, 
											cv2.FONT_HERSHEY_SIMPLEX, 
											fontScale, 2)
			# put filled text rectangle
			cv2.rectangle(image_np_with_detections, (xmin, ymin), 
							(xmin + text_width, 
							ymin - text_height - baseline), 
							bbox_color, thickness=cv2.FILLED)
			# put text above rectangle
			cv2.putText(image_np_with_detections, label_text, 
							(xmin, ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
							fontScale, (255, 255, 0), bbox_thick, 
							lineType=cv2.LINE_AA)

			# Draw bounding box
			cv2.rectangle(image_np_with_detections, 
					(xmin, ymin), 
					(xmax, ymax), 
					(bbox_color), bbox_thick*2)


			# if results are provided append
			# the detections results
			if bbox_results is not None:
				bbox_results.add_new_result(width, height, label, scores[i], [xmin, xmax, ymin, ymax])

	if (valid_detections == 0) and (bbox_results is not None):
		bbox_results.add_new_result(width, height, "N/D", "N/D", ["N/D", "N/D", "N/D", "N/D"])

	if save_results:
		saving_start = time.time()
		print("[IMAGEUTILS] Saving results...")

		cv2.imwrite('results.png', image_np_with_detections)
		print("[IMAGEUTILS] Results saved succesfully, took {} ms"
			.format((time.time()-saving_start)*1000))

	total_drawing = (time.time()-total_drawing_start)*1000
	print("[IMAGEUTILS] Total drawing pipeline took {} ms"
			.format(total_drawing))
	
	return total_drawing

def draw_yolo_bounding_boxes(image, results, labels, save=False, bbox_results=None):
	print("[IMAGEUTILS] Starting drawing pipeline...")
	start_drawing = time.time()

	print("[IMAGEUTILS] Setting class, shape and colors...")
	# extract info of classes and image
	num_classes = len(labels)
	height, width, _ = image.shape

	# set random colors to draw
	hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
	colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
	colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

	random.seed(0)
	random.shuffle(colors)
	random.seed(None)

	valid_detections = 0

	print("[IMAGEUTILS] Now drawing results...")
	results_start = time.time()
	boxes, scores, classes, num_boxes = results
	for i in range(num_boxes[0]):
		if int(classes[0][i]) < 0 or int(classes[0][i]) > num_classes: continue
		
		valid_detections += 1
		box = boxes[0][i]
		box[0] = int(box[0] * height)
		box[2] = int(box[2] * height)
		box[1] = int(box[1] * width)
		box[3] = int(box[3] * width)

		score = scores[0][i]
		class_ind = int(classes[0][i])
		label = labels[class_ind]

		# set some display parameters
		bbox_thick = int(0.6 * (height + width) / 1000)
		if bbox_thick < 1: bbox_thick = 1
		fontScale = 0.75 * bbox_thick

		bbox_color = colors[class_ind]
		c1, c2 = (box[1], box[0]), (box[3], box[2])
		cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

		# set label display parameters
		label_text = '%s: %.2f' % (label, score)
		t_size = cv2.getTextSize(label_text, 0, fontScale, thickness=bbox_thick // 2)[0]
		c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
		cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled

		cv2.putText(image, label_text, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
					fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

		# if results are provided append
		# the detections results
		if bbox_results is not None:
			bbox_results.add_new_result(width, height, label, score, [box[1], box[3], box[0], box[2]])

	if (valid_detections == 0) and (bbox_results is not None):
		bbox_results.add_new_result(width, height, "N/D", "N/D", ["N/D", "N/D", "N/D", "N/D"])

	cv2.imshow("Yolo detector" ,image)
	if cv2.waitKey(1) == ord('q'):
		cv2.destroyAllWindows()
		sys.exit()

	print("[IMAGEUTILS] Finished drawing results, it took {}"
			.format((time.time()-results_start)*1000))
	
	total_drawing = (time.time()-start_drawing)*1000
	print("[IMAGEUTILS] Drawing pipeline finished in {} ms"
			.format(total_drawing))

	if (valid_detections == 0):
		print("[IMAGEUTILS] THERE WAS NO DETECTIONS HERE!")

	if save:
		cv2.imwrite("yolo-results.png", image)

	return total_drawing

def draw_tracker_info(frame, label, tracker):
	height, width, _ = frame.shape
	
	# display vertical boundaries
	left_bound = tracker.boundaries[0]
	mid_bound = tracker.boundaries[1]

	cv2.line(frame, (left_bound, int(height*0.07)), (left_bound, height), (0, 0, 255), 4)
	cv2.line(frame, (mid_bound, int(height*0.07)), (mid_bound, height), (0, 255, 255), 1)
	
	# display total objects counted and active objects
	total_objects_text = "TOTAL {}S: {}".format(label, tracker.total_trackers)
	active_objects_text = "ACTIVE {}S: {}".format(label,tracker.active_trackers)
	cv2.putText(frame, total_objects_text, (int(width*0.1+width*0.06), int(height*0.05)), 
										cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3) # Draw label text
	cv2.putText(frame, active_objects_text, (int(width*0.1+width*0.28), int(height*0.05)), 
										cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 128, 255), 3) # Draw label text

	return frame

def display_frame(frame, scaled=False, message="Detections"):
	frame = frame.copy()
	if scaled:
		monitor = get_monitors()[0]
		VIDEO_WIDTH = int(monitor.width - monitor.width * 0.035) # Take a litle space for the window frame
		VIDEO_HEIGHT = int(monitor.height - monitor.height * 0.06)
		scaled_frame = resize_image(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))

	else:
		scaled_frame = frame.copy()

	cv2.imshow(message, scaled_frame)
	cv2.moveWindow(message, 0, 0)

	if cv2.waitKey(1) == ord('q'):
		cv2.destroyAllWindows()
		sys.exit()

def save_frame(frame, output_path, detections_path, frame_count, folder="detections_frames"):
	last_path_of_detections_path = os.path.basename(os.path.normpath(detections_path))
	splitted_detections_path = last_path_of_detections_path.split("_")

	model_name = "_".join(splitted_detections_path[0:2])
	precision = splitted_detections_path[2]
	threshold = splitted_detections_path[3]
	frame_dims = splitted_detections_path[6]
	loading_backend = splitted_detections_path[7].split(".")[0]
	
	if output_path[-1] != '/':
		output_path+='/'

	output_path += "{}/{}-{}-{}-{}-{}-frames/".format(folder, model_name, precision, threshold, frame_dims, loading_backend)
	create_dir(output_path)
 
	frame_output_path = "{}000{}.png".format(output_path, frame_count)
	print("[IMGUTILS] Saving frame {} to {}...".format(frame_count, frame_output_path))
	start_saving=time.time()
	cv2.imwrite(frame_output_path, frame)
	total_saving= (time.time()-start_saving)*1000
	print("[IMGUTILS] Done!, it took {}ms".format(total_saving))
	
def create_dir(output_path):
	    # creating the output directory that will contain the model
    print("[DIR-CREATION] Attempting to create output directory...")
    print("[DIR-CREATION] Model will be saved under {}"
        .format(output_path))
    if not os.path.exists(output_path):
        print("[DIR-CREATION] Directory doesnt exists, creating...") 
        os.makedirs(output_path)
        print("[DIR-CREATION] Directory created succesfully...") 
    else:
        print("[DIR-CREATION] Output directory already exists...")

def preprocess_yolo_image(self, image, input_size):
	print("[INFERENCE] Preprocessing image (normalization and resizing)...")
	preprocess_start = time.time()

	processed_image = cv2.resize(image, (input_size, input_size))
	processed_image = processed_image/255
	processed_image = processed_image[np.newaxis, ...].astype(np.float32)

	print("[INFERENCE] Image preprocessed, it took {} ms"
			.format((time.time()-preprocess_start)*1000))

	return processed_image
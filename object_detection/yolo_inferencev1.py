import cv2
import tensorflow as tf
import numpy as np
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from results_utils import BboxResults, YoloPerformanceResults

import glob, re
import time
import image_utils as imgutils

from yolo_core import utils

class YoloInferator():
    def __init__(self, verbose=1):
        self.attributes = {}

    def get_func_from_saved_model(self, saved_model_dir):
        """ Gets the graph function to perform inference
            from a saved .pb model """
        
        if "FP16" in saved_model_dir or "fp16" in saved_model_dir:
            print("[MODEL] Model precision recognized as FP16")
            self.attributes["precision"] = "FP16"
        elif "INT8" in saved_model_dir or "in8" in saved_model_dir:
            print("[MODEL] Model precision recognized as INT8")
            self.attributes["precision"] = "INT8"
        elif "FP32" in saved_model_dir or "fp32" in saved_model_dir:
            print("[MODEL] Model precision recognized as FP32")
            self.attributes["precision"] = "FP32"
        else:
            print("[MODEL] Model precision unknown")
            self.attributes["precision"] = "UNKNOWN"

        print("[MODEL] Loading saved model from {}...".format(saved_model_dir))
        model_start = time.time()
        # Load the saved model
        saved_model_loaded = tf.saved_model.load(
            saved_model_dir, tags=[tag_constants.SERVING])

        print("[MODEL] Succesfully loaded saved model from {}...".format(saved_model_dir))
        print("[MODEL] Model loaded in {} ms".format((time.time()-model_start)*1000))

        print("[MODEL] Creating graph function from model...")
        graph_func_time = time.time()
        # Creates a graph function from the saved model
        #graph_func = saved_model_loaded.signatures[
        #    list(saved_model_loaded.signatures.keys())[0]]

        print("[MODEL] Succesfully created graph func from model from model")
        print("[MODEL] Graph func loading took {} ms"
                .format((time.time()-graph_func_time)*1000))

        
        self.attributes["saved_model_loaded"] = saved_model_loaded
        self.attributes["graph_func"] = saved_model_loaded.signatures['serving_default']
        return saved_model_loaded.signatures['serving_default']

    def config_gpu_memory(self, gpu_mem_cap):
        """ Function to set the memory usage of the gpu to perform inference"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if not gpus:
            return
        print('[GPU-CONFIG] Found the following GPUs:')
        for gpu in gpus:
            print('[GPU-CONFIG]  ', gpu)
        for gpu in gpus:
            try:
                if not gpu_mem_cap:
                    # Allow for memory to growth as requires
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print("[GPU-CONFIG] Configured to allow memory growth")
                else:
                    # Explicitly states a memory limit
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=gpu_mem_cap)])
                    print("[GPU-CONFIG] Configured max memory limit of to {}"
                            .format(gpu_mem_cap))
            except RuntimeError as e:
                print('[GPU-CONFIG] Can not set GPU memory config', e)

    def load_model_details(self, labels_path, is_tiny, model_dir):
        print("[MODEL] Loading model details...")
        details_start = time.time()
        strides, anchors, num_class, xyscale = utils.load_config(labels_path, is_tiny)
        self.attributes['STRIDES'] = strides
        self.attributes['ANCHORS'] = anchors
        self.attributes['NUM_CLASS'] = num_class
        self.attributes['XYSCALE'] = xyscale
        self.attributes['LABELS'] = utils.read_class_names(labels_path)

        details_path = model_dir + 'details.txt'
        with open(details_path, 'r') as f:
            for line in f.readlines():
                splitted = line.split('=')
                key = splitted[0]
                value = splitted[1]

                if value[-1] == "\n":
                    #print("[MODEL] Repairing key {} with value {}".format(key, value))
                    value = value[0:-1]

                print("[MODEL] Adding key {} with value {} to attributes"
                    .format(key, value))
                self.attributes[key] = value
                #print("[MODEL] Key loaded? {}".format(self.attributes[key]))

        print("[MODEL] Model details loaded, it took {} ms"
                .format((time.time()-details_start)*1000))

    def run_inference(self, image_path=None, warmup_iters=0, threshold=0.5, iou=0.45, model_dir=None, bbox_results=None, performance_results=None, opencv=False, resize=False):   
        input_size = int(self.attributes["INPUT_SIZE"])
        labels = self.attributes["LABELS"]
        # get a copy of the graph func
        #graph_func = self.attributes["graph_func"]
        # Load the saved model
        model_loaded = tf.saved_model.load(
            model_dir, tags=[tag_constants.SERVING])
        saved_model_loaded = model_loaded.signatures['serving_default']

        #warmup
        if warmup_iters > 0:
            print("[INFERENCE] Starting warmup on {} iterations..."
                    .format(warmup_iters))
            warmup_start = time.time()
            # get input size from model attributes
            input_size = int(self.attributes["INPUT_SIZE"])
    
            # create a set of random images and perform inference
            for i in range(warmup_iters):
                print("[INFERENCE] Generating image {} with dims {}x{}"
                    .format(i+1, input_size, input_size))

                # working with numpy/cv backend
                if opencv:
                    # create random image
                    resized_image = np.random.normal(size=(input_size, input_size, 3)).astype(np.float32)/255.

                    # conventional conversion (use with opencv option)
                    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
                    #input_tensor = tf.convert_to_tensor(resized_image)
                    #input_tensor = tf.convert_to_tensor(image)
                    # The model expects a batch of images, so add an axis with `tf.newaxis`.
                    #input_tensor = input_tensor[tf.newaxis, ...]

                    images_data = []
                    for i in range(1):
                        images_data.append(resized_image)
                    images_data = np.asarray(images_data).astype(np.float32)
                    input_tensor = tf.constant(images_data)

                # working with tf backend for image
                else:
                    dataset, _, _ = imgutils.get_dataset_tf(tensor_type='float', input_size=input_size)
                    """print("[WARMUP] Creating features...")
                    features = np.random.normal(loc=112, scale=70,
                            size=(1, input_size, input_size, 3)).astype(np.float32)
                    features = np.clip(features, 0.0, 255.0).astype(np.float32)
                    features = tf.convert_to_tensor(value=tf.compat.v1.get_variable(
                                        "features", initializer=tf.constant(features)))
                    print("[WARMUP] Creating dataset from features...")
                    dataset = tf.data.Dataset.from_tensor_slices([features])
                    dataset = dataset.repeat(count=1)"""
                    
                    print("[WARMUP] Retrieving image and input tensor...")
                    dataset_enum = enumerate(dataset)
                    # get input tensor and cast to image (np)
                    input_tensor = list(dataset_enum)[0][1]
                    resized_image = input_tensor.numpy()[0]
                    
                    images_data = []
                    for i in range(1):
                        images_data.append(resized_image)
                    images_data = np.asarray(images_data).astype(np.float32)
                    input_tensor = tf.constant(images_data)

                
                # get the detections
                print("[WARMUP] Now performing warmup inference...")
                inference_start_time = time.time()
                detections = saved_model_loaded(input_tensor)
                print("[WARMUP] Warmup inference took {} ms"
                        .format((time.time()-inference_start_time)*1000))

                print("[INFERENCE] Preprocessing network outputs...")
                start_output = time.time()
                # extract output tensors metadata: boxes, confidence scores
                print("[INFERENCE] Extracting output tensors metadata...")
                keyval_start = time.time()
                for key, value in detections.items():
                    #print("[INFERENCE-DEBUG] key {}: value {}".format(key, value))
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]
                    #print("[INFERENCE-DEBUG] boxes ", boxes, type(boxes))
                    #print("[INFERENCE-DEBUG] confidence ", pred_conf, type(pred_conf))

                print("[INFERENCE] Done extracting metadata, it took {}"
                        .format((time.time()-keyval_start)*1000))

                print("[INFERENCE] Performing NMS to output...")
                nms_start = time.time()
                # perform non-max supression to retrieve valid detections only
                boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                    boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                    scores=tf.reshape(
                        pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                    max_output_size_per_class=50,
                    max_total_size=50,
                    iou_threshold=iou,
                    score_threshold=threshold
                )

                results = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
                print("[INFERENCE] NMS done, it took {} ms"
                        .format((time.time()-nms_start)*1000))
                total_output = (time.time()-start_output)*1000
                print("[INFERENCE] Done procesing output!, it took {}ms".format(total_output))
            
                _ = imgutils.draw_yolo_bounding_boxes(resized_image, results, labels)
                                # display results in ms
            print("[WARMUP] Warmup finished, it took {} ms"
                    .format((time.time()-warmup_start)*1000))

        # case inference
        if opencv:
            print("[INFERENCE] Loading image with opencv backend...")
            # opencv option
            image_loading_start = time.time()
            image = imgutils.read_image_from_cv2(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            total_image_loading = (time.time()-image_loading_start)*1000

            # preprocess image to work on tf
            print("[INFERENCE] Preprocessing image...")
            start_preprocessing = time.time()

            # resize image to netwrk input dimensions
            resized_image = imgutils.resize_image(image, (input_size, input_size))
            resized_image = resized_image/255.

            # conventional conversion (use with opencv option)
            # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
            #input_tensor = tf.convert_to_tensor(resized_image)
            #input_tensor = tf.convert_to_tensor(image)
            # The model expects a batch of images, so add an axis with `tf.newaxis`.
            #input_tensor = input_tensor[tf.newaxis, ...]

            images_data = []
            for i in range(1):
                images_data.append(resized_image)
            images_data = np.asarray(images_data).astype(np.float32)
            input_tensor = tf.constant(images_data)

            total_preprocessing = (time.time()-start_preprocessing)*1000
            print("[INFERENCE] Preprocessing done!, it took {}ms"
                .format(total_preprocessing))
            

        # case tf backend to manipulate images
        else:
            print("[INFERENCE] Loading image with tf backend...")
            # dataset option, yolo models require resizing to input size
            dataset, total_image_loading, total_preprocessing = imgutils.get_dataset_tf(image_path=image_path, 
                                                                                        input_size=input_size,
                                                                                        tensor_type='float')
            #print("[INFERENCE] dataset {}".format(dataset))
            # take the batched image
            dataset_enum = enumerate(dataset)
            input_tensor = list(dataset_enum)[0][1]
            
            # take image as np and convert to rgb
            image_bgr = input_tensor.numpy()[0]
            resized_image = image_bgr[...,::-1].copy()/255.
            
            images_data = []
            for i in range(1):
                images_data.append(resized_image)
            images_data = np.asarray(images_data).astype(np.float32)
            input_tensor = tf.constant(images_data)

            print("[INFERENCE] Images data: {} - shape: {} - dtype {} - type {}"    
                .format(images_data, images_data.shape, images_data.dtype, type(images_data)))

            print("[INFERENCE] Input tensor: {} - shape: {} - dtype {} - type {}"
                .format(input_tensor, input_tensor.shape, input_tensor.dtype, type(input_tensor)))
            

        print("[INFERENCE] Now performing inference...")
        inference_start_time = time.time()

        #print("[INFERENCE] Input tensor: {} - dtype {}"
        #    .format(input_tensor, input_tensor.dtype))
            
        # get the detections
        detections = saved_model_loaded(input_tensor)

        total_inference = (time.time()-inference_start_time)*1000
        print("[INFERENCE] Inference took {} ms"
            .format(total_inference))

        print("[INFERENCE] Preprocessing network outputs...")
        start_output = time.time()
        # extract output tensors metadata: boxes, confidence scores
        print("[INFERENCE] Extracting output tensors metadata...")
        keyval_start = time.time()
        for key, value in detections.items():
            #print("[INFERENCE-DEBUG] key {}: value {}".format(key, value))
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]
            #print("[INFERENCE-DEBUG] boxes ", boxes, type(boxes))
            #print("[INFERENCE-DEBUG] confidence ", pred_conf, type(pred_conf))

        print("[INFERENCE] Done extracting metadata, it took {}"
                .format((time.time()-keyval_start)*1000))

        print("[INFERENCE] Performing NMS to output...")
        nms_start = time.time()
        # perform non-max supression to retrieve valid detections only       
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iou,
            score_threshold=threshold
        )

        results = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        print("[INFERENCE] NMS done, it took {} ms"
                .format((time.time()-nms_start)*1000))
        total_output = (time.time()-start_output)*1000
        print("[INFERENCE] Done procesing output!, it took {}ms".format(total_output))

        # draw results on image
        if (bbox_results is not None) and (performance_results is not None):
            drawing_time = imgutils.draw_yolo_bounding_boxes(resized_image, results, labels,
                                                                bbox_results=bbox_results)
            
            height, width, _ = image.shape
            # add new performance results to object
            performance_results.add_new_result(width, height, total_image_loading, 
                                            total_preprocessing, total_inference, 
                                            total_output, drawing_time)
        else:
            _ = imgutils.draw_yolo_bounding_boxes(resized_image, results, labels, save=True)
    
        # debugging info
        """print("[INFERENCE-DEBUG] tf-boxes ", boxes, type(boxes))
        print("[INFERENCE-DEBUG] tf-scores", scores, type(scores))
        print("[INFERENCE-DEBUG] tf-classes", classes, type(classes))
        print("[INFERENCE-DEBUG] tf-valid-detections", valid_detections, type(valid_detections))
        
        print("[INFERENCE-DEBUG] nmsed-boxes ", results[0])
        print("[INFERENCE-DEBUG] nmsed-scores ", results[1])
        print("[INFERENCE-DEBUG] nmsed-classes ", results[2])
        print("[INFERENCE-DEBUG] nmsed-valid-detections ", results[3])"""

    def run_inference_on_images(self, images_path, warmup_iters=0, model_dir=None, labels=None, threshold=0.3, iou=0.45, opencv=False):
        input_size = int(self.attributes["INPUT_SIZE"])
        labels = self.attributes["LABELS"]

        # get a copy of the graph func
        saved_model_loaded = tf.saved_model.load(
            model_dir, tags=[tag_constants.SERVING])
        saved_model_loaded = saved_model_loaded.signatures['serving_default']

        #warmup
        if warmup_iters > 0:
            print("[INFERENCE] Starting warmup on {} iterations..."
                    .format(warmup_iters))
            warmup_start = time.time()
            # get input size from model attributes
            input_size = int(self.attributes["INPUT_SIZE"])
    
            # create a set of random images and perform inference
            for i in range(warmup_iters):
                print("[INFERENCE] Generating image {} with dims {}x{}"
                    .format(i+1, input_size, input_size))

                # working with numpy/cv backend
                if opencv:
                    # create random image
                    image = np.random.normal(size=(input_size, input_size, 3)).astype(np.float32)/255.
                    # conventional conversion (use with opencv option)
                    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
                    # resize image to netwrk input dimensions
                    resized_image = imgutils.resize_image(image, (input_size, input_size))/255.

                    images_data = []
                    for i in range(1):
                        images_data.append(resized_image)
                    images_data = np.asarray(images_data).astype(np.float32)
                    input_tensor = tf.constant(images_data)

                    print("[INFERENCE] Images data: {} - shape: {} - dtype {} - type {}"
                        .format(images_data, images_data.shape, images_data.dtype, type(images_data)))

                    print("[INFERENCE] Input tensor: {} - shape: {} - dtype {} - type {}"
                        .format(input_tensor, input_tensor.shape, input_tensor.dtype, type(input_tensor)))

                # working with tf backend for image
                else:
                    dataset, _, _ = imgutils.get_dataset_tf(tensor_type='float', input_size=input_size)
                    
                    print("[WARMUP] Retrieving image and input tensor...")
                    dataset_enum = enumerate(dataset)
                    # get input tensor and cast to image (np)
                    input_tensor = list(dataset_enum)[0][1]
                    
                    # take image as np and convert to rgb
                    image_bgr = input_tensor.numpy()[0]
                    resized_image = image_bgr[...,::-1].copy()/255.

                    images_data = []
                    for i in range(1):
                        images_data.append(resized_image)
                    images_data = np.asarray(images_data).astype(np.float32)
                    input_tensor = tf.constant(images_data)

                    print("[INFERENCE] Images data: {} - shape: {} - dtype {} - type {}"
                        .format(images_data, images_data.shape, images_data.dtype, type(images_data)))

                    print("[INFERENCE] Input tensor: {} - shape: {} - dtype {} - type {}"
                        .format(input_tensor, input_tensor.shape, input_tensor.dtype, type(input_tensor)))

                # get the detections
                print("[WARMUP] Now performing warmup inference...")
                inference_start_time = time.time()
                detections = saved_model_loaded(input_tensor)
                print("[WARMUP] Warmup inference took {} ms"
                        .format((time.time()-inference_start_time)*1000))

                print("[INFERENCE] Preprocessing network outputs...")
                start_output = time.time()
                # extract output tensors metadata: boxes, confidence scores
                print("[INFERENCE] Extracting output tensors metadata...")
                keyval_start = time.time()
                for key, value in detections.items():
                    #print("[INFERENCE-DEBUG] key {}: value {}".format(key, value))
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]
                    #print("[INFERENCE-DEBUG] boxes ", boxes, type(boxes))
                    #print("[INFERENCE-DEBUG] confidence ", pred_conf, type(pred_conf))

                print("[INFERENCE] Done extracting metadata, it took {}"
                        .format((time.time()-keyval_start)*1000))

                print("[INFERENCE] Performing NMS to output...")
                nms_start = time.time()
                # perform non-max supression to retrieve valid detections only
                boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                    boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                    scores=tf.reshape(
                        pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                    max_output_size_per_class=50,
                    max_total_size=50,
                    iou_threshold=iou,
                    score_threshold=threshold
                )

                results = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
                print("[INFERENCE] NMS done, it took {} ms"
                        .format((time.time()-nms_start)*1000))
                total_output = (time.time()-start_output)*1000
                print("[INFERENCE] Done procesing output!, it took {}ms".format(total_output))
            
                _ = imgutils.draw_yolo_bounding_boxes(resized_image, results, labels)
                                # display results in ms
            print("[WARMUP] Warmup finished, it took {} ms"
                    .format((time.time()-warmup_start)*1000))

        # define frame count and a helper function to read
        # the images sorted by numerical index
        frame_count = 1
        numbers = re.compile(r'(\d+)')
        def numericalSort(value):
            parts = numbers.split(value)
            parts[1::2] = map(int, parts[1::2])
            return parts
        
        # read image sorted by numerical order
        image_paths = sorted(glob.glob(images_path + "*.png"), key=numericalSort)
        print("[INFERENCE] Found {} images in {} ...".format(len(image_paths), images_path))

        #create a class to store results
        bbox_results = BboxResults()
        performance_results = YoloPerformanceResults()

        # Iterate over all images, perform inference and update 
        # results dataframe
        for image_path in image_paths:
            print("[INFERENCE] Processing frame/image {} from {}".format(frame_count, image_path))
            image_filename = image_path.split('/')[-1]

            # init metadata
            bbox_results.init_frame_metadata(image_filename, frame_count)
            performance_results.init_frame_metadata(image_filename, frame_count)

            # case inference
            if opencv:
                print("[INFERENCE] Loading image with opencv backend...")
                # opencv option
                image, total_image_loading = imgutils.load_image_as_np(image_path, "CV2")

                image = image.astype(np.float32)

                # preprocess image to work on tf
                print("[INFERENCE] Preprocessing image...")
                start_preprocessing = time.time()

                # resize image to netwrk input dimensions
                resized_image = imgutils.resize_image(image, (input_size, input_size))/255.

                images_data = []
                for i in range(1):
                    images_data.append(resized_image)
                images_data = np.asarray(images_data).astype(np.float32)
                input_tensor = tf.constant(images_data)

                print("[INFERENCE] Images data: {} - shape: {} - dtype {} - type {}"
                    .format(images_data, images_data.shape, images_data.dtype, type(images_data)))

                print("[INFERENCE] Input tensor: {} - shape: {} - dtype {} - type {}"
                    .format(input_tensor, input_tensor.shape, input_tensor.dtype, type(input_tensor)))

                total_preprocessing = (time.time()-start_preprocessing)*1000
                print("[INFERENCE] Preprocessing done!, it took {}ms"
                    .format(total_preprocessing))
                

            # case tf backend to manipulate images
            else:
                print("[INFERENCE] Loading image with tf backend...")
                # dataset option, yolo models require resizing to input size
                dataset, total_image_loading, total_preprocessing = imgutils.get_dataset_tf(image_path=image_path, 
                                                                                            input_size=input_size,
                                                                                            tensor_type='float')
                #print("[INFERENCE] dataset {}".format(dataset))
                # take the batched image
                dataset_enum = enumerate(dataset)

                local_preprocessing_start = time.time()
                input_tensor = list(dataset_enum)[0][1]
                # take image as np and convert to rgb
                image_bgr = input_tensor.numpy()[0]
                resized_image = image_bgr[...,::-1].copy()

                images_data = []
                for i in range(1):
                    images_data.append(resized_image)
                images_data = np.asarray(images_data).astype(np.float32)
                input_tensor = tf.constant(images_data)

                print("[INFERENCE] Images data: {} - shape: {} - dtype {} - type {}"
                    .format(images_data, images_data.shape, images_data.dtype, type(images_data)))

                print("[INFERENCE] Input tensor: {} - shape: {} - dtype {} - type {}"
                    .format(input_tensor, input_tensor.shape, input_tensor.dtype, type(input_tensor)))
                total_local_preprocessing = (time.time()-local_preprocessing_start)*1000

                total_preprocessing += total_local_preprocessing

            print("[INFERENCE] Now performing inference...")
            inference_start_time = time.time()
            #print("[INFERENCE] Input tensor: {} - dtype {}"
            #    .format(input_tensor, input_tensor.dtype))
                
            # get the detections
            detections = saved_model_loaded(input_tensor)

            total_inference = (time.time()-inference_start_time)*1000
            print("[INFERENCE] Inference took {} ms"
                .format(total_inference))

            print("[INFERENCE] Preprocessing network outputs...")
            start_output = time.time()
            # extract output tensors metadata: boxes, confidence scores
            print("[INFERENCE] Extracting output tensors metadata...")
            keyval_start = time.time()
            for key, value in detections.items():
                #print("[INFERENCE-DEBUG] key {}: value {}".format(key, value))
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]
                #print("[INFERENCE-DEBUG] boxes ", boxes, type(boxes))
                #print("[INFERENCE-DEBUG] confidence ", pred_conf, type(pred_conf))

            print("[INFERENCE] Done extracting metadata, it took {}"
                    .format((time.time()-keyval_start)*1000))

            print("[INFERENCE] Performing NMS to output...")
            nms_start = time.time()
            # perform non-max supression to retrieve valid detections only
            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=iou,
                score_threshold=threshold
            )

            results = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
            print("[INFERENCE] NMS done, it took {} ms"
                    .format((time.time()-nms_start)*1000))
            total_output = (time.time()-start_output)*1000
            print("[INFERENCE] Done procesing output!, it took {}ms".format(total_output))

            # TODO: ADD tracker updating 

            # draw results on image
            if (bbox_results is not None) and (performance_results is not None):
                drawing_time = imgutils.draw_yolo_bounding_boxes(resized_image, results, labels,
                                                                    bbox_results=bbox_results)
                
                height, width, _ = image.shape
                # add new performance results to object
                performance_results.add_new_result(width, height, total_image_loading, 
                                                total_preprocessing, total_inference, 
                                                total_output, drawing_time)
            else:
                _ = imgutils.draw_yolo_bounding_boxes(resized_image, results, labels, save=True)

            print("[INFERENCE] Image/frame {} processed".format(frame_count))
            frame_count += 1

        print("[INFERENCE] All frames procesed!")

        output_path = "{}results/{}".format(images_path, self.attributes["MODEL_NAME"])
        if opencv:
            output_path += "/opencv-backend"
        else:
            output_path += "/tf-backend"
        
        model_name = self.attributes["MODEL_NAME"]
        precision = self.attributes["precision"]
        
        # save results obtained from performance and detections to output
        bbox_results.save_results(output_path, model_name, precision, 
                                threshold, resize=True, opencv=opencv)
        performance_results.save_results(output_path, model_name, precision, 
                                threshold, resize=True, opencv=opencv)

            





    

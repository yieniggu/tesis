import cv2
import tensorflow as tf
import numpy as np
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

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
    
    def load_model_details(self, labels_path, is_tiny, input_size):
        print("[MODEL] Loading model details...")
        details_start = time.time()
        strides, anchors, num_class, xyscale = utils.load_config(labels_path, is_tiny)
        self.attributes['STRIDES'] = strides
        self.attributes['ANCHORS'] = anchors
        self.attributes['NUM_CLASS'] = num_class
        self.attributes['XYSCALE'] = xyscale
        self.attributes['INPUT_SIZE'] = input_size
        self.attributes['LABELS'] = utils.read_class_names(labels_path)

        print("[MODEL] Model details loaded, it took {} ms"
                .format((time.time()-details_start)*1000))

    def run_inference(self, image_path=None, warmup_iters=0, threshold=0.5, iou=0.45, image=None, model_dir=None):   
        # get a copy of the graph func
        #graph_func = self.attributes["graph_func"]
        if model_dir is not None:
            saved_model_loaded = self.get_func_from_saved_model(model_dir)
        else: 
            saved_model_loaded = self.attributes["saved_model_loaded"]
        
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

                # create random image
                image = np.random.randint(low = 0, high = 255, 
                        size = (input_size, input_size, 3)).astype('uint8')
                
                # preprocess image to work on tf
                print("[INFERENCE] Converting warmup image to tf constant as batch data...")
                image_to_tensor_start = time.time()
                image_data = self.preprocess_image(image, self.attributes['INPUT_SIZE'])
                # Convert the input to a tf constant to perform inference.
                batch_data = tf.constant(image_data)
                
                print("[INFERENCE] Conversion took {} ms"
                    .format((time.time()-image_to_tensor_start)*1000))

                # get the detections
                print("[INFERENCE] Now performing warmup inference...")
                inference_start_time = time.time()
                pred_bbox = saved_model_loaded(batch_data)
                print("[INFERENCE] Warmup inference took {} ms"
                    .format((time.time()-inference_start_time)*1000))
            
            # display results in ms
            print("[INFERENCE] Warmup finished, it took {} seconds"
                    .format(time.time()-warmup_start))

        # actual inference
        print("[INFERENCE] Loading image...")
        # Load image
        if image_path is not None:
            image = imgutils.load_image_as_np(image_path, "CV2")

        image_data = self.preprocess_image(image, self.attributes['INPUT_SIZE'])
        
        # preprocess image to work on tf
        print("[INFERENCE] Converting image to tf constant as batch data...")
        image_to_tensor_start = time.time()
        # Convert the input to a tf constant to perform inference.
        batch_data = tf.constant(image_data)
        
        print("[INFERENCE] Conversion took {} ms"
            .format((time.time()-image_to_tensor_start)*1000))

        # get the detections
        print("[INFERENCE] Now performing inference...")
        inference_start_time = time.time()
        pred_bbox = saved_model_loaded(batch_data)
        print("[INFERENCE] Inference took {} ms"
            .format((time.time()-inference_start_time)*1000))
        #print("[INFERENCE-DEBUG] Output tensor ", pred_bbox, type(pred_bbox))

        # extract output tensors metadata: boxes, confidence scores
        print("[INFERENCE] Extracting output tensors metadata...")
        keyval_start = time.time()
        for key, value in pred_bbox.items():
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
        
        # debugging info
        """print("[INFERENCE-DEBUG] tf-boxes ", boxes, type(boxes))
        print("[INFERENCE-DEBUG] tf-scores", scores, type(scores))
        print("[INFERENCE-DEBUG] tf-classes", classes, type(classes))
        print("[INFERENCE-DEBUG] tf-valid-detections", valid_detections, type(valid_detections))
        """
        print("[INFERENCE-DEBUG] nmsed-boxes ", results[0])
        print("[INFERENCE-DEBUG] nmsed-scores ", results[1])
        print("[INFERENCE-DEBUG] nmsed-classes ", results[2])
        print("[INFERENCE-DEBUG] nmsed-valid-detections ", results[3])

 
        imgutils.draw_yolo_bounding_boxes(image=image, results=results, labels=self.attributes['LABELS'])

    def preprocess_image(self, image, input_size):
        print("[INFERENCE] Preprocessing image (normalization and resizing)...")
        preprocess_start = time.time()

        processed_image = cv2.resize(image, (input_size, input_size))
        processed_image = processed_image/255
        processed_image = processed_image[np.newaxis, ...].astype(np.float32)

        print("[INFERENCE] Image preprocessed, it took {} ms"
                .format((time.time()-preprocess_start)*1000))

        return processed_image




    

import pandas as pd
import os

def add_element(temp, element, element_message, class_message):
    print("[{}] Adding {} {} to {}"
        .format(class_message.upper(), element_message,   
               element, class_message))
    temp.append(element)

def add_results(temp, results, class_message):
    print("[{}] Adding temp data to {}"
        .format(class_message.upper(), class_message))
    #print(temp)
    results.append(temp)
    #print("results: ", results)

def init_common_metadata(bbox_results, performance_results, image_path, frame_count):
    bbox_results.init_frame_metadata(image_path, frame_count)
    performance_results.init_frame_metadata(image_path, frame_count)

def save_common_results(bbox_results, performance_results, output_path, model_name, precision_mode, threshold, resize=False, opencv=False):
    bbox_results.save_results(output_path, model_name, precision_mode, threshold, resize, opencv)
    performance_results.save_results(output_path, model_name, precision_mode, threshold, resize, opencv)

def create_dir(output_path):
	    # creating the output directory that will contain the model
    print("[DIR-CREATION] Attempting to create output directory...")
    print("[DIR-CREATION] Files will be saved under {}"
        .format(output_path))
    if not os.path.exists(output_path):
        print("[DIR-CREATION] Directory doesnt exists, creating...") 
        os.makedirs(output_path)
        print("[DIR-CREATION] Directory created succesfully...") 
    else:
        print("[DIR-CREATION] Output directory already exists...")

class BboxResults:
    def __init__(self, results=[], temp=[]):
        print("[BBOX-RESULTS] Initializing class with values " +
                "{} for results and {} for temp".format(results, temp))
        self.results = results
        self.temp = temp
        self.class_message = "bbox-results"
        self.current_frame = None
        self.current_image_path = None

    def init_frame_metadata(self, image_path, frame_count):
        self.image_path = image_path
        self.current_frame = frame_count

    def add_image_path(self):
        add_element(self.temp, self.image_path, "image", self.class_message)
    
    def add_frame_count(self):
        add_element(self.temp, self.current_frame, "frame", self.class_message)

    def add_width(self, width):
        add_element(self.temp, width, "width", self.class_message)

    def add_height(self, height):
        add_element(self.temp, height, "height", self.class_message)

    def add_class(self, detection_class):
        add_element(self.temp, detection_class, "class", self.class_message)

    def add_score(self, score):
        add_element(self.temp, score, "score", self.class_message)

    def add_bbox(self, bbox):
        add_element(self.temp, bbox[0], "xmin", self.class_message)
        add_element(self.temp, bbox[1], "xmax", self.class_message)
        add_element(self.temp, bbox[2], "ymin", self.class_message)
        add_element(self.temp, bbox[3], "ymax", self.class_message)

    def add_new_result(self, width, height, detection_class, score, bbox):
        self.add_image_path()
        self.add_frame_count()
        self.add_width(width)
        self.add_height(height)
        self.add_class(detection_class)
        self.add_score(score)
        self.add_bbox(bbox)
        self.reset()

    def save_results(self, output_path, model_name, precision_mode, threshold, resize=False, opencv=False):
        if output_path[-1] != "/":
            output_path += "/"
        
        output_path += "detections/"
        create_dir(output_path)
        
        save_output = "{}{}_{}_{}_detection_results".format(output_path, model_name, 
                                                            precision_mode, threshold)

        
        if resize:
            save_output += "_resized"
        else:
            save_output += "_original"
            
        if opencv:
            save_output += "_opencv"
        else:
            save_output += "_tf"


        results_df = pd.DataFrame(self.results, 
                                columns=["filename", "frame", "width", "height", "detection_class",
                                        "detection_score", "xmin", "xmax",
                                        "ymin", "ymax"])

        print("[BBOX-RESULTS] Saving detection results to {}".format(save_output))
        results_df.to_excel(save_output+ ".xlsx", index=False)
        results_df.to_csv(save_output+ ".csv", index=False)
        print("[BBOX-RESULTS] Results saved succesfully!")

    def reset(self):
        add_results(self.temp, self.results, self.class_message)
        self.temp = []

class PerformanceResults:
    def __init__(self, results=[], temp=[]):
        print("[PERFORMANCE-RESULTS] Initializing class with values " +
            "{} for results and {} for temp".format(results, temp))
        self.results = results
        self.temp = temp
        self.class_message = "performance-results"
        self.current_frame = None
        self.current_image_path = None


    def init_frame_metadata(self, image_path, frame_count):
        self.image_path = image_path
        self.current_frame = frame_count

    def add_image_path(self):
        add_element(self.temp, self.image_path, "image", self.class_message)
    
    def add_frame_count(self):
        add_element(self.temp, self.current_frame, "frame", self.class_message)

    def add_width(self, width):
        add_element(self.temp, width, "width", self.class_message)

    def add_height(self, height):
        add_element(self.temp, height, "height", self.class_message)

    def add_inference_time(self, inference_time):
        add_element(self.temp, inference_time, "inference time", self.class_message)
    
    def add_drawing_time(self, drawing_time):
        add_element(self.temp, drawing_time, "drawing time", self.class_message)

    def add_image_loading_time(self, image_loading_time):
        add_element(self.temp, image_loading_time, "image loading time", self.class_message)

    def add_conversion_time(self, conversion_time):
        add_element(self.temp, conversion_time, "conversion time", self.class_message)

    def add_new_result(self, width, height, image_loading_time, conversion_time, inference_time, drawing_time):
        self.add_image_path()
        self.add_frame_count()
        self.add_width(width)
        self.add_height(height)
        self.add_image_loading_time(image_loading_time)
        self.add_conversion_time(conversion_time)
        self.add_inference_time(inference_time)
        self.add_drawing_time(drawing_time)
        self.reset()

    def save_results(self, output_path, model_name, precision_mode, threshold, resize=False, opencv=False):
        if output_path[-1] != "/":
            output_path += "/"
        
        output_path += "performance/"
        create_dir(output_path)
        save_output = "{}{}_{}_{}_performance_results".format(output_path, model_name, 
                                                            precision_mode, threshold)

        if resize:
            save_output += "_resized"
        else:
            save_output += "_original"
            
        if opencv:
            save_output += "_opencv"
        else:
            save_output += "_tf"

        results_df = pd.DataFrame(self.results, 
                                columns=["filename", "frame", "width", "height", "image_loading_time",
                                        "preprocessing_time", "inference_time", "drawing_time"])

        print("[PERFORMANCE-RESULTS] Saving performance results to {}".format(save_output))
        results_df.to_excel(save_output+ ".xlsx", index=False)
        results_df.to_csv(save_output+ ".csv", index=False)
        print("[PERFORMANCE-RESULTS] Results saved succesfully!")

    def reset(self):
        add_results(self.temp, self.results, self.class_message)
        self.temp = []

    
class SortResults:
    def __init__(self, results=[], temp=[]):
        print("[SORT-RESULTS] Initializing class with values " +
            "{} for results and {} for temp".format(results, temp))
        self.results = results
        self.temp = temp
        self.class_message = "sort-results"
        self.current_frame = None
        self.current_image_path = None

    def init_frame_metadata(self, image_path, frame_count):
        self.image_path = image_path
        self.current_frame = frame_count

    def add_frame_path(self):
        add_element(self.temp, self.image_path, "image", self.class_message)
    
    def add_frame_count(self):
        add_element(self.temp, self.current_frame, "frame", self.class_message)

    def add_width(self, width):
        add_element(self.temp, width, "width", self.class_message)

    def add_height(self, height):
        add_element(self.temp, height, "height", self.class_message)

    def add_tracker_id(self, tracker_id):
        add_element(self.temp, tracker_id, "tracker_id", self.class_message)

    def add_tracker_state(self, state):
        add_element(self.temp, state, "state", self.class_message)

    def add_tracker_time_since_update(self, time_since_update):
        add_element(self.temp, time_since_update, "time since update", self.class_message)

    def add_initialization(self, initialized_in_roi):
        add_element(self.temp, initialized_in_roi, "intialization", self.class_message)

    def add_tracker_centroid(self, centroid):
        add_element(self.temp, centroid[0], "centroid_x", self.class_message)
        add_element(self.temp, centroid[1], "centroid_y", self.class_message)

    def add_bbox(self, bbox):
        add_element(self.temp, bbox[0], "xmin", self.class_message)
        add_element(self.temp, bbox[1], "xmax", self.class_message)
        add_element(self.temp, bbox[2], "ymin", self.class_message)
        add_element(self.temp, bbox[3], "ymax", self.class_message)

    def reset(self):
        add_results(self.temp, self.results, self.class_message)
        self.temp = []

    def add_new_result(self, width, height, tracker_id, tracker_state, time_since_update, initialized_in_roi, first_centroid, last_centroid, bbox):
        self.add_frame_path()
        self.add_frame_count()
        self.add_width(width)
        self.add_height(height)
        self.add_tracker_id(tracker_id)
        self.add_tracker_state(tracker_state)
        self.add_tracker_time_since_update(time_since_update)
        self.add_initialization(initialized_in_roi)
        self.add_tracker_centroid(first_centroid)
        self.add_tracker_centroid(last_centroid)
        self.add_bbox(bbox)
        self.reset()

    def save_results(self, output_path, detections_path):
        last_path_of_detections_path = os.path.basename(os.path.normpath(detections_path))
        splitted_detections_path = last_path_of_detections_path.split("_")

        model_name = "_".join(splitted_detections_path[0:-6])
        precision = splitted_detections_path[-6]
        threshold = splitted_detections_path[-5]
        frame_dims = splitted_detections_path[-2]
        loading_backend = splitted_detections_path[-1].split(".")[0]

        if output_path[-1] != '/':
            output_path+= '/'
        output_path += "tracker_results/tracking/"

        create_dir(output_path)

        output_path += "{}_{}_{}_sort-tracking_results_{}_{}".format(model_name, precision, threshold, frame_dims, loading_backend)

        results_df = pd.DataFrame(self.results, columns=["filename", "frame_count", "width", "height",
                                                    "tracker_id", "tracker_state", 
                                                    "time_since_update", "init_in_roi", 
                                                    "first_centroid_x", "first_centroid_y",
                                                    "last_centroid_x", "last_centroid_y",
                                                    "xmin", "xmax", "ymin", "ymax"])

        print("[SORT-RESULTS] Saving results to {}".format(output_path))
        results_df.to_excel(output_path + ".xlsx", index=False)
        results_df.to_csv(output_path + ".csv", index=False)
        print("[SORT-RESULTS] Results saved!")
        

class SortPerformanceResults:
    def __init__(self, results=[], temp=[]):
        print("[SORT-Performance] Initializing class with values " +
            "{} for results and {} for temp".format(results, temp))
        self.results = results
        self.temp = temp
        self.class_message = "sort-performance"
        self.current_frame = None
        self.current_image_path = None

    def init_frame_metadata(self, image_path, frame_count):
        self.image_path = image_path
        self.current_frame = frame_count

    def add_frame_path(self):
        add_element(self.temp, self.image_path, "image", self.class_message)

    def add_frame_count(self):
        add_element(self.temp, self.current_frame, "frame", self.class_message)

    def add_width(self, width):
        add_element(self.temp, width, "width", self.class_message)

    def add_height(self, height):
        add_element(self.temp, height, "height", self.class_message)

    def add_image_loading_time(self, image_loading_time):
        add_element(self.temp, image_loading_time, "image loading time", self.class_message)

    def add_image_preprocessing(self, image_preprocessing):
        add_element(self.temp, image_preprocessing, "image preprocesing time", self.class_message)

    def add_preparing_time(self, preparing_time):
        add_element(self.temp, preparing_time, "preparing time", self.class_message)

    def add_update_time(self, update_time):
        add_element(self.temp, update_time, "tracker update time", self.class_message)

    def add_tracker_results(self, tracker_results_time):
        add_element(self.temp, tracker_results_time, "tracker results time", self.class_message)

    def add_update_state_time(self, update_state_time):
        add_element(self.temp, update_state_time, "update state time", self.class_message)

    def add_total_trackers(self, total_trackers):
        add_element(self.temp, total_trackers, "total_trackers", self.class_message)

    def reset(self):
        add_results(self.temp, self.results, self.class_message)
        self.temp = []

    def add_new_result(self, width, height, image_loading_time, image_preprocessing, preparing_time, update_time, results_time, update_state_time, total_trackers):
        self.add_frame_path()
        self.add_frame_count()
        self.add_width(width)
        self.add_height(height)
        self.add_image_loading_time(image_loading_time)
        self.add_image_preprocessing(image_preprocessing)
        self.add_preparing_time(preparing_time)
        self.add_update_time(update_time)
        self.add_tracker_results(results_time)
        self.add_update_state_time(update_state_time)
        self.add_total_trackers(total_trackers)
        self.reset()

    def save_results(self, output_path, detections_path):
        last_path_of_detections_path = os.path.basename(os.path.normpath(detections_path))
        splitted_detections_path = last_path_of_detections_path.split("_")

        model_name = "_".join(splitted_detections_path[0:-6])
        precision = splitted_detections_path[-6]
        threshold = splitted_detections_path[-5]
        frame_dims = splitted_detections_path[-2]
        loading_backend = splitted_detections_path[-1].split(".")[0]

        if output_path[-1] != '/':
            output_path+= '/'
        
        output_path += "tracker_results/performance/"

        create_dir(output_path)

        output_path += "{}_{}_{}_sort-performance_results_{}_{}".format(model_name, precision, threshold, frame_dims, loading_backend)

        results_df = pd.DataFrame(self.results, columns=["filename", "frame_count", "width", "height",
                                                    "image_loading_time", "preprocessing_time",
                                                    "preparing_time", "update_time", "results_time",
                                                    "update_state_time", "total_trackers"])

        print("[SORT-PERFORMANCE] Saving results to {}".format(output_path))
        results_df.to_excel(output_path + ".xlsx", index=False)
        results_df.to_csv(output_path + ".csv", index=False)
        print("[SORT-PERFORMANCE] Results saved!")
        
class YoloPerformanceResults:
    def __init__(self, results=[], temp=[]):
        print("[YOLO-PERFORMANCE-RESULTS] Initializing class with values " +
            "{} for results and {} for temp".format(results, temp))
        self.results = results
        self.temp = temp
        self.class_message = "yolo-performance-results"
        self.current_frame = None
        self.current_image_path = None


    def init_frame_metadata(self, image_path, frame_count):
        self.image_path = image_path
        self.current_frame = frame_count

    def add_image_path(self):
        add_element(self.temp, self.image_path, "image", self.class_message)
    
    def add_frame_count(self):
        add_element(self.temp, self.current_frame, "frame", self.class_message)

    def add_width(self, width):
        add_element(self.temp, width, "width", self.class_message)

    def add_height(self, height):
        add_element(self.temp, height, "height", self.class_message)

    def add_inference_time(self, inference_time):
        add_element(self.temp, inference_time, "inference time", self.class_message)
    
    def add_drawing_time(self, drawing_time):
        add_element(self.temp, drawing_time, "drawing time", self.class_message)

    def add_image_loading_time(self, image_loading_time):
        add_element(self.temp, image_loading_time, "image loading time", self.class_message)

    def add_conversion_time(self, conversion_time):
        add_element(self.temp, conversion_time, "conversion time", self.class_message)

    def add_output_processing_time(self, output_processing_time):
        add_element(self.temp, output_processing_time, "output processing time", self.class_message)

    def add_new_result(self, width, height, image_loading_time, conversion_time, inference_time, output_processing_time, drawing_time):
        self.add_image_path()
        self.add_frame_count()
        self.add_width(width)
        self.add_height(height)
        self.add_image_loading_time(image_loading_time)
        self.add_conversion_time(conversion_time)
        self.add_inference_time(inference_time)
        self.add_output_processing_time(output_processing_time)
        self.add_drawing_time(drawing_time)
        self.reset()

    def save_results(self, output_path, model_name, precision_mode, threshold, resize=False, opencv=False):
        if output_path[-1] != "/":
            output_path += "/"
        
        output_path += "performance/"
        create_dir(output_path)
        save_output = "{}{}_{}_{}_performance_results".format(output_path, model_name, 
                                                            precision_mode, threshold)

        if resize:
            save_output += "_resized"
        else:
            save_output += "_original"
            
        if opencv:
            save_output += "_opencv"
        else:
            save_output += "_tf"

        results_df = pd.DataFrame(self.results, 
                                columns=["filename", "frame", "width", "height", "image_loading_time",
                                        "preprocessing_time", "inference_time", "output_processing_time",
                                        "drawing_time"])

        print("[YOLO-PERFORMANCE-RESULTS] Saving performance results to {}".format(save_output))
        results_df.to_excel(save_output+ ".xlsx", index=False)
        results_df.to_csv(save_output+ ".csv", index=False)
        print("[YOLO-PERFORMANCE-RESULTS] Results saved succesfully!")

    def reset(self):
        add_results(self.temp, self.results, self.class_message)
        self.temp = []
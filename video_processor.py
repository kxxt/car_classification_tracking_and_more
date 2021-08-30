import cv2
import numpy as np
import onnxruntime as onnx
import json
from PIL import Image, ImageDraw, ImageFont
from optparse import OptionParser
import matplotlib.pyplot as plt
from math import tan
import pandas as pd

# Parse the command line arguments.
parser = OptionParser()
parser.add_option("-i", "--input", type="string", dest="input",
                  help="input file", metavar="FILE")
parser.add_option("-o", "--output", type="string", dest="output",
                  help="output file", metavar="FILE", default="output.mp4")
parser.add_option("-t", "--draw-tracker-bbox",
                  action="store_true", dest="draw_tracker_bbox", default=False,
                  help="Draw the tracker bbox on the processed video.")
parser.add_option("-e", "--interactive",
                  action="store_true", dest="interactive", default=False,
                  help="Show the processed video in realtime and visualize the trackers when inited, etc.")
parser.add_option("-s", "--silent",
                  action="store_true", dest="silent", default=False,
                  help="Hide all cmdline outputs.")
parser.add_option("-l", "--license-recognition-result", type="string", dest="license_recognition_result_file",
                  default="license_recognition_result.csv", metavar="FILE",
                  help="The result csv file of license plate recognition.")
parser.add_option("-d", "--dump",
                  action="store_true", dest="dump", default=False,
                  help="Dump every frame's analysis data into result.json.")

# Define the color palette.
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 255, 255), (0, 0, 0), (255, 128, 128),
    (128, 255, 128), (128, 128, 255), (255, 128, 0),
    (255, 0, 128), (0, 255, 128), (128, 255, 0),
    (0, 128, 255), (128, 0, 255)
]


def get_iou(a, b, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width < 0) or (height < 0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou


# A simple function to check if a character is a chinese character
is_chinese = lambda ch: '\u4e00' <= ch <= '\u9fff'


def tlbr2xywh(tl, br):
    """
    Convert the top-left, bottom-right format bbox to the top-left, width, height format
    :param tl: top-left point (x,y)
    :param br: bottom-right point (x,y)
    :return: top-left, width, height: (x,y,w,h)
    """
    return tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]


def cv2ImgAddText(img, text, left, top, text_color=(0, 255, 0), text_size=24):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    font = ImageFont.truetype("simhei.ttf", text_size, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, text_color, stroke_width=1, font=font)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# Specify the type of the tracker that will be used in vehicle tracking
new_tracker = cv2.TrackerCSRT_create


class TrackerWrapper:
    """
    Wraps the opencv tracker.
    """

    def __init__(self, vehicle_id):
        self.tracker = new_tracker()  # inner real opencv tracker
        self.failure_cnt = 0  # How many frames did the tracker failed
        self.last_bbox = None  # The last tracker bbox
        self.vehicle_id = vehicle_id  # The id of the vehicle which the tracker tracks
        self.lifetime = 0  # How many frames the tracker lived
        self.frame = None  # Last frame used to update `last_bbox`
        self.detected = False  # Is the tracker detected in the current frame
        self.category = None  # The category of the vehicle
        self.license = None  # The license plate of the vehicle
        self.last_license_bbox = None
        # if it's not detected, this field should be None

    def plot(self):
        """
        Plot the tracker using matplotlib.
        """
        plt.imshow(self.roi)
        plt.show()  # Call the show method to explicitly show the plot

    @property
    def roi(self):
        """
        Get the region of the tracker.
        :return: numpy ndarray
        """
        bbox = self.last_bbox
        img = self.frame
        if bbox is None or img is None:
            # If any information is missing, just return None
            return None
        # Sometimes the tracker's predicted bbox is out of the whole image.
        # So when the pos is negative, we replace it with zero.
        top = bbox[1] if bbox[1] > 0 else 0
        left = bbox[0] if bbox[0] > 0 else 0
        return img[top:top + bbox[3], left:left + bbox[2]]

    def __getattr__(self, item):
        """
        Here we use the deep black magic of meta programming to wrap the opencv tracker.
        By doing so, we can attach some methods and properties/fields on it.

                                                                          -- kxxt, sduwh
        """
        return self.tracker.__getattribute__(item)

    def __repr__(self):
        """
        Make it pretty! Debugger is your friend after you can pretty print annoying objects.
        """
        return f"<Tracker {self.vehicle_id}:{self.category if self.category is not None else '?'}," \
               f" bbox={self.last_bbox}, failure_cnt={self.failure_cnt}, lifetime={self.lifetime}>"

    def dict(self):
        """
        Get the dict representation of the tracker's current status.
        :return: dict
        """
        return {
            "last_bbox": self.last_bbox,
            "license": self.license,
            "vehicle_id": self.vehicle_id,
            "category": self.category,
            "failure_cnt": self.failure_cnt,
            "detected": self.detected,
            "last_license_bbox": self.last_license_bbox
        }


def init_new_tracker(img, bbox, vehicle_id):
    """
    Init new tracker with the region of img specified by the bbox.
    :param vehicle_id: the unique id of the vehicle
    :param img: the whole frame
    :param bbox: the region of the frame
    :return: the initialized brand-new tracker.
    """
    tracker = TrackerWrapper(vehicle_id)
    tracker.init(img, bbox)  # Init the opencv tracker
    tracker.frame = img  # Store current frame
    tracker.detected = True  # Mark the tracker as detected since this function is called from a detection
    # Plot the tracker. remove "-e" in the cmdline if you get bothered.
    if options.interactive:
        plt.imshow(img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]])
        plt.show()
    return tracker


def init_session(confidence_lowerbound=0.53, font=cv2.FONT_HERSHEY_SIMPLEX):
    """
    Init the video processing session.
    :param confidence_lowerbound: the confidence lower bound for car detection.
    :param font: the font used in cv2.putText
    :return: a function which takes a frame and returns the processed frame.
    """
    # Configure onnx runtime options
    opts = onnx.SessionOptions()
    opts.log_severity_level = 3  # Disable some annoying onnxruntime logs
    # Load our inference models
    car_detection_sess = onnx.InferenceSession("models/car_detection_model_fixed.onnx", opts)
    car_classification_sess = onnx.InferenceSession("models/car_classification_model.onnx", opts)
    # Check to see if we are on a GPU device.
    # If not, please run `pip install onnxruntime-gpu`
    print(f"Using {onnx.get_device()}")
    # Get the input tensor names for our models
    inp_car_detection = car_detection_sess.get_inputs()[0].name
    inp_car_classification = car_classification_sess.get_inputs()[0].name

    # Load the label <=> integer mapping
    with open("models/car_classification_model.labels") as f:
        labels = json.load(f)
    # Map integers to labels.
    num2label = {labels[key]: key for key in labels}

    # Store current trackers.
    trackers = []

    # when the tracker's failure_cnt is more than this threshold, we drop the tracker.
    tracking_failure_count_upperbound = 5

    # Threshold decide whether the tracker bbox and the detection bbox is the same one.
    IoU_threshold = 0.40

    def pixel_coordinate_transform(y, param):
        """
        Transform the pixel coordinate to the real distance.
        The unit of the real distance is "meter".
        :param y: the pixel coordinate y.
        :param param: the list of 4 paramaters [A, B, C, D] of transform
        """
        A, B, C, D = param
        return (tan((y - D) / A) - C) / B

    def update_trackers(img):
        """
        Update all the trackers in the list.
        :param img: the input frame.
        :return: None
        """
        to_be_removed = []  # Store the bad trackers
        # Update every tracker.
        for index, tracker in enumerate(trackers):
            # bbox: array of x,y,w,h
            success, bbox = tracker.update(img)
            tracker.lifetime += 1  # Increase each tracker's lifetime
            tracker.detected = False  # Clear the detected status because we're starting at a new frame
            if success:
                # If the tracker succeeded, we store the frame and the last_bbox in the tracker.
                tracker.last_bbox = bbox
                tracker.frame = img
            else:
                # If the tracker failed, we clear the last_bbox and frame fields and increase the failure counter
                tracker.last_bbox = None
                tracker.last_license_bbox = None
                tracker.frame = None
                tracker.failure_cnt += 1
                if tracker.failure_cnt > tracking_failure_count_upperbound:
                    # If a tracker fails too many times, it's not a good tracker. Just delete it.
                    to_be_removed.append(tracker)

        # Remove the bad trackers.
        for tracker in to_be_removed:
            print(f"Remove tracker {tracker.vehicle_id} due to too many failures.")
            trackers.remove(tracker)

        # Notify us about the count of trackers we have.
        print(f"There are {len(trackers)} trackers now.")

    def preprocess_image(img):
        """
        Preprocess image for vehicle detection.
        :param img:
        :return: preprocess image
        """
        return np.transpose(
            np.array(
                cv2.resize(img, (800, 800)),  # Resize the img to 800 * 800
                dtype="float32"),  # Make it float, make it float!
            (2, 0, 1)  # transpose the ndarray from (255, 255, 3) to (3, 255, 255)
        )[np.newaxis, :] / 255  # add a new axis and normalize the image!

    def preprocess_vehicle_region(img):
        """
        Preprocess vehicle region for vehicle classification.
        :param img: vehicle region
        :return: preprocess image
        """
        return np.transpose(
            np.array(
                cv2.resize(img, (224, 224)),  # Resize the img to 224 * 224
                dtype="float32"),  # Make it float, make it float!
            (2, 0, 1)  # transpose the ndarray from (255, 255, 3) to (3, 255, 255)
        )[np.newaxis, :] / 255  # add a new axis and normalize the image!

    def box_transform(box, h, w):
        """
        Transform a box of range [0,1]*[0,1] to [0,w]*[0,h].
        :param box: indexable of 4 floats
        :param h: integer
        :param w: integer
        :return: Tuple[Tuple[int, int], Tuple[int,int]]
        """
        return (int(box[0] * w), int(box[1] * h)), (int(box[2] * w), int(box[3] * h))

    # vehicle id counter
    counter = 0

    def generate_car_id():
        """
        Generate unique vehicle id.
        :return: vehicle id of integer
        """
        nonlocal counter
        # The code is trivial and do not need comments
        counter += 1
        return counter

    def draw_detection(img, box, license_recognition_data, detection_frame=None, transform_bbox=True):
        """
        Draw detection box on img.
        :param license_recognition_data: this frame's license_recognition data, in pandas dataframe format.
        :param detection_frame: frame to perform detection
        :param transform_bbox: transform_bbox before using.
        :param img: frame to draw
        :param box: detection bbox
        :return: the tracker of the vehicle and the consumed license plate record ids and the drawn frame and the detection result dict
        """

        if transform_bbox:
            # Let the transform do the magic!
            tl, br = box_transform(box, img.shape[0], img.shape[1])
        else:
            # Keep the box unchanged.
            tl = (box[0], box[1])
            br = (box[2], box[3])

        # Storage the ious and corresponding indices
        ious = []
        for index, tracker in enumerate(trackers):
            if tracker.last_bbox is not None:
                # if the tracker get something we are interested in, then dig into it.
                bbox = tracker.last_bbox
                ious.append(
                    (
                        index,  # Store the id since we will use it.
                        get_iou(  # Just calculate the IoU of detection and tracking.
                            (tl[0], tl[1], br[0], br[1]),
                            (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
                        )
                    )
                )
        indices = []  # Store indices whose corresponding bbox is similar to our detection bbox
        for index, iou in ious:
            if iou > IoU_threshold:
                indices.append(index)  # It's apparent
        if indices:  # If we get something
            # Search for the tracker with the least id,
            # it's the first tracker that get applied to the corresponding vehicle
            min_id_index = min(indices, key=lambda x: trackers[x].vehicle_id)
            min_id_tracker = trackers[min_id_index]
            # Search for the tracker with the biggest IoU,
            # It should replace others.
            max_iou_index = ious[max(range(len(ious)), key=lambda x: ious[x][1])][0]
            max_iou_tracker = trackers[max_iou_index]

            # Replace max IoU tracker's some fields with the min id tracker's fields,
            # because the min id tracker comes first. It knows more about the car.
            max_iou_tracker.vehicle_id = min_id_tracker.vehicle_id
            max_iou_tracker.license = min_id_tracker.license

            # The max_iou_tracker is the tracker that we want to give back to the caller.
            ret_tracker = max_iou_tracker
            if max_iou_tracker.detected:
                # If the tracker is already detected in another detection,
                # then this detection is redundant.
                # We do not need to do anything.
                print(f"Redundant detection found! tracker: {max_iou_tracker}")
                return ret_tracker, set(), img, None
            else:
                # Mark the max IoU tracker as detected.
                max_iou_tracker.detected = True

            # Remove trackers with smaller IoU
            trackers_to_be_removed = [trackers[x] for x in indices if x != max_iou_index]
            for tracker in trackers_to_be_removed:
                print(f"Remove tracker {tracker.vehicle_id} due to small iou!")
                trackers.remove(tracker)

            vehicle_id = ret_tracker.vehicle_id  # Store the result vehicle id
        else:
            # We didn't find any matched tracker bbox.
            # It's probably a new vehicle!
            print(f"No iou > IoU_threshold!")
            if br[1] > 900 or br[1] < 100:
                # When the vehicle is at bottom or top, do not track it.
                # If we do track it, many issues do rise due to the partial car image!
                print(f"Currently not tracking {tl},{br} since it's at bottom or top")
                # The -1 id means the vehicle is detected but not currently tracked
                vehicle_id = -1
                ret_tracker = None  # not currently tracking, return None to the caller
            else:
                # We find a new car! Give it an id!
                vehicle_id = generate_car_id()
                # Tracking it by adding a new tracker!
                ret_tracker = init_new_tracker(detection_frame, tlbr2xywh(tl, br), vehicle_id)
                trackers.append(ret_tracker)
                print(f"Detected new vehicle! id:{vehicle_id}")

        # Make it colorful and draw the detection bbox!
        cv2.rectangle(img, tl, br, colors[vehicle_id % len(colors)], 2)

        consumed_license_plates = set()  # Store the consumed license plates' indices
        if ret_tracker is not None:  # If we are tracking something
            # Iterate through all available license plates data in this frame
            for index, record in license_recognition_data.iterrows():
                lpbbox = (record['left'], record['top'], record['w'], record['h'])  # The bbox of the license plate
                lptext = record['license']  # The text on the license plate
                if lpbbox[0] > tl[0] and lpbbox[1] > tl[1] \
                        and lpbbox[0] + lpbbox[2] < br[0] and lpbbox[1] + lpbbox[3] < br[1]:
                    if lptext != 'None':
                        # if the license plate bbox is inside the vehicle region
                        if ret_tracker.license is not None and ret_tracker.license != lptext:
                            # if the detected license is different from the license stored in the tracker
                            print(f"License of {ret_tracker} is inconsistent! {ret_tracker.license}, new:{lptext}!")
                            if is_chinese(lptext[0]) and not is_chinese(ret_tracker.license[0]):
                                # The license plate text should begin with a chinese character.
                                ret_tracker.license = lptext
                            else:
                                # If both text begins with a chinese character, we take the longer text.
                                ret_tracker.license = max((ret_tracker.license, lptext), key=lambda x: len(x))
                        else:
                            # If we didn't know the license text before, we store it in the tracker
                            ret_tracker.license = lptext
                    ret_tracker.last_license_bbox = lpbbox
                    consumed_license_plates.add(index)  # Mark the license plate data as consumed.
                    break  # Just find the first one, not need to check other data.
            if ret_tracker.last_license_bbox is None: print("No license plate detected!")
            if ret_tracker.license is not None:
                # If we have the car's license plate, write it on the bottom left corner of the bbox.
                img = cv2ImgAddText(img, ret_tracker.license, tl[0] + 5, br[1] - 30)

        # Draw the vehicle id on the top right corner of the bbox!
        cv2.putText(img, str(vehicle_id), (br[0] - 30 * len(str(vehicle_id)), tl[1] + 34), font, 1.2,
                    colors[vehicle_id % len(colors)], 2, cv2.LINE_AA)

        detection_dict = {  # If `-d` specified, dump the detection in a dict
            "tl": tl,
            "br": br,
            "vehicle_id": vehicle_id,
            "tracker": ret_tracker
        } if options.dump else None
        return ret_tracker, consumed_license_plates, img, detection_dict

    def draw_tracker(img, tracker):
        """
        Draw tracker bbox on img
        :param tracker: tracker to visualize
        :param img: frame to draw on
        :return: None
        """
        bbox = tracker.last_bbox
        if bbox is None:
            return
        # Draw the tracker bbox
        color = (255, 255, 255) if tracker.vehicle_id is None else colors[tracker.vehicle_id % len(colors)]
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2)
        if tracker.vehicle_id is not None:
            # Draw the vehicle id of the tracker
            cv2.putText(img, str(tracker.vehicle_id), (bbox[0], bbox[1] + 30), font, 1.2, color, 1,
                        cv2.LINE_AA)

    def draw_classification(img, detection_frame, box, tracker=None, transform_bbox=True):
        """
        Using detection_frame and box to classify a vehicle and draw the classification result on img
        :param img: frame to draw on
        :param detection_frame: frame that contains the vehicle
        :param box: bbox of the vehicle
        :return: None
        """
        if transform_bbox:
            tl, br = box_transform(box, detection_frame.shape[0], detection_frame.shape[1])
        else:
            tl = (box[0], box[1])
            br = (box[2], box[3])
        if tracker is None or tracker.category is None:
            vehicle_region = detection_frame[tl[1]:br[1], tl[0]:br[0]]
            # Transform the vehicle region so that it fits the model.
            transformed = preprocess_vehicle_region(vehicle_region)
            # Get the probabilities
            probs = car_classification_sess.run(None, {inp_car_classification: transformed})[0]
            # The most probable category id!
            max_prob_id = np.argmax(probs)
            # Get the category by the num <=> label mapping.
            category = num2label[max_prob_id]
            if tracker is not None:
                tracker.category = category
        else:
            category = tracker.category
        # Draw the category on the video.
        cv2.putText(img, category, (tl[0], tl[1] - 10), font, 1.2, (255, 33, 100), 2, cv2.LINE_AA)

    def process(detection_frame, license_plate_data):
        # transform the frame for detection.
        transformed = preprocess_image(detection_frame)
        # Copy the frame to draw on it. (Do not mix up the draw frame and the detection frame.)
        draw_frame = detection_frame.copy()
        # Get the results from model.
        boxes, classes, confs = car_detection_sess.run(None, {inp_car_detection: transformed})
        normalized_boxes = (box / 800 for box in boxes)  # Normalize the bboxes
        infos = zip(normalized_boxes, confs)  # Zip normalized boxes and their confidence values.
        update_trackers(detection_frame)  # Update all the trackers.
        useless_trackers = [tracker for tracker in trackers  # collect useless trackers
                            if tracker.roi is not None and
                            # We think a tracker useless when its region is almost of the same color.
                            # The 20 threshold is just a value from my experience.
                            np.std(tracker.roi) < 32]
        useless_trackers.extend(tracker for tracker in trackers if
                                tracker.last_bbox is not None and
                                # Area too small, this kind of tracker should be removed.
                                tracker.last_bbox[2] * tracker.last_bbox[3] < 1000)
        for tracker in useless_trackers:
            trackers.remove(tracker)
            print(f"Remove {tracker} because it's std or area is too small.")
        consumed_license_plates = set()  # Store the indices of the consumed license plates
        detections = []  # Store the detection dicts
        for info in infos:
            if info[1] < confidence_lowerbound:
                # We do not draw the detection which is below our confidence lower bound.
                # Just break it, because the array is sorted!
                break
            # Get the tracker, consumed license plate indices , processed frame, detection dict
            tracker, consumed_lps, draw_frame, detection_dict = draw_detection(draw_frame, info[0], license_plate_data,
                                                                               detection_frame)
            consumed_license_plates.update(consumed_lps)  # Update consumed license plate indices with consumed_lps
            draw_classification(draw_frame, detection_frame, info[0], tracker)  # Draw the classification of the vehicle
            if detection_dict is not None:
                detection_dict['tracker'] = detection_dict["tracker"].dict() \
                    if detection_dict["tracker"] is not None else None  # Dump the tracker as a dict.
                detections.append(detection_dict)  # Append this detection dict to the list of detections

        to_be_removed = []
        for tracker in trackers:
            if tracker.last_bbox is not None and not tracker.detected:
                print(f"Undetected tracker: {tracker}!")
                roi = tracker.roi
                std = np.std(roi)  # Calculate the standard deviation of the roi,
                yl = roi.shape[0]  # lower area of the roi and mid area of the roi.
                lower_area = roi[yl // 2:yl, :, :]
                mid_area = roi[yl // 4:3 * yl // 4, :, :]
                mid_area_std = np.std(mid_area)
                if std < 35:
                    # If the standard deviation is too small,
                    # the information contained in the roi is considered too little.
                    # This kind of trackers are removed here.
                    to_be_removed.append(tracker)
                elif std > 45 and np.std(lower_area) > 40:  # If the roi contains enough information
                    if mid_area_std < 39.98 or tracker.last_bbox[1] < -20 or tracker.last_bbox[0] < -20:
                        # We no longer track a car when it is near the edge of the image.
                        # In this situation, the tracker usually go out of bounds and
                        # the vehicle region is just a part of the whole vehicle.
                        # Thus it's easy to detect this situation and remove corresponding trackers.
                        to_be_removed.append(tracker)

                    if roi.shape[0] * roi.shape[1] < 4400:
                        # The area is too small thus it's probably not a vehicle
                        to_be_removed.append(tracker)
                    # Recover the detection box
                    bbox = tracker.last_bbox

                    # Make sure the bbox is within the frame
                    bbox = (0 if bbox[0] < 0 else bbox[0], 0 if bbox[1] < 0 else bbox[1], bbox[2], bbox[3])

                    # Convert the xywh format to x1x2y1y2 format
                    x1y1x2y2_bbox = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])

                    # Draw the detection. Since we are using the real bbox, not the normalized ones,
                    # we should pass `transform_bbox=False` to the `draw_detection` function.
                    _, consumed_lps, draw_frame, detection_dict = draw_detection(draw_frame, x1y1x2y2_bbox,
                                                                                 license_plate_data,
                                                                                 detection_frame, transform_bbox=False)
                    consumed_license_plates.update(consumed_lps)
                    draw_classification(draw_frame, detection_frame, x1y1x2y2_bbox, tracker, transform_bbox=False)
                    if detection_dict is not None:
                        detection_dict['tracker'] = detection_dict["tracker"].dict() \
                            if detection_dict["tracker"] is not None else None
                        detections.append(detection_dict)

        license_plates = []  # Store dumped license plates
        for index, record in license_plate_data.iterrows():
            # Draw the license plate bbox
            cv2.rectangle(draw_frame, (int(record["left"]), int(record['top'])),
                          (int(record['left'] + record['w']), int(record['top'] + record['h'])), (0, 255, 0), 2)
            if options.dump:  # Append the license plate data in a dict format
                license_plates.append({
                    "left": record['left'],
                    "top": record['top'],
                    "width": record['w'],
                    "height": record['h'],
                    "license": record['license'] if record['license'] != "None" else None,
                    "score": record['score'],
                    "used": index in consumed_license_plates
                })
            if index not in consumed_license_plates and record['license'] != "None":
                # We lost a detection, or we are not tracking this vehicle
                print(f"undetected car's license plate: {record['license']}")
                # In this situation, directly draw the license plate text under the license plate bbox
                draw_frame = cv2ImgAddText(draw_frame, record['license'], record['left'],
                                           record['top'] + record['h'] + 1)
        for tracker in to_be_removed:  # Perform the removal
            trackers.remove(tracker)
        if options.draw_tracker_bbox:
            # if `-t` specified, draw the tracker bbox on the frame.
            for tracker in trackers:
                draw_tracker(draw_frame, tracker)  # Visualize all the good trackers。
        return draw_frame, {
            "detections": detections,
            "license_plates": license_plates
        }

    return process


if __name__ == "__main__":
    # Parse cmdline arguments
    (options, args) = parser.parse_args()
    if not options.input:
        # Missing input file, stop execution.
        parser.error("Missing input file!")
    if options.silent:
        # Override the print function to make the whole script silent
        print = lambda x: None
    # Init video
    cap = cv2.VideoCapture(options.input)
    result_file = options.output
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    video_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(result_file, video_fourcc, fps_video, (frame_width, frame_height))

    # Init video processing session to get the function which processes frames.
    process = init_session()

    frame_id = 0  # Store the frame id
    # Load license recognition result data.
    df = pd.read_csv(options.license_recognition_result_file)

    dump_result = []  # Store analysis results of frames

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            # We reached the end of the video.
            print("Finished!")
            break
        # Pass the frame and license plate data of that frame to the function
        draw_frame, frame_analysis = process(frame, df[df['frame_id'] == frame_id + 1])
        if options.dump:
            dump_result.append(frame_analysis | {"frame_id": frame_id})
        # Write the processed frame
        video_writer.write(draw_frame)
        if options.interactive:
            # If we are in interactive mode, show the video to user.
            cv2.imshow("Video", draw_frame)
            cv2.waitKey(20)
        frame_id += 1  # Increase the frame id counter
    video_writer.release()  # Release the video file since we've finished processing.
    if options.dump:
        with open("result.json", "w") as f:  # Save the analysis result to `result.json`
            json.dump({
                "width": frame_width,
                "height": frame_height,
                "fps": fps_video,
                "frames": dump_result
            }, f)
    print(f"Processed {frame_id + 1} frames in total!")

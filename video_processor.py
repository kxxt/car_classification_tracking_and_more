import cv2
import numpy as np
import onnxruntime as onnx
import json
from license_recognition import license_recognition
from optparse import OptionParser
from functools import reduce
import matplotlib.pyplot as plt

# Parse the command line arguments.
parser = OptionParser()
parser.add_option("-i", "--input", type="string", dest="input",
                  help="input file", metavar="FILE")
parser.add_option("-o", "--output", type="string", dest="output",
                  help="output file", metavar="FILE", default="output.mp4")
parser.add_option("-n", "--no-license-recognition",
                  action="store_false", dest="license_recognition", default=True,
                  help="Do not perform license recognition during processing.")
parser.add_option("-e", "--interactive",
                  action="store_true", dest="interactive", default=False,
                  help="Show the processed video in realtime and visualize the trackers when inited, etc.")

# Define the color palette.
colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 255),
    (0, 0, 0),
    (255, 128, 128),
    (128, 255, 128),
    (128, 128, 255),
    (255, 128, 0),
    (255, 0, 128),
    (0, 255, 128),
    (128, 255, 0),
    (0, 128, 255),
    (128, 0, 255)
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


def tlbr2xywh(tl, br):
    return tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]

    # Decide the type of tracker and store trackers in a list.


new_tracker = cv2.TrackerCSRT_create


class TrackerWrapper:
    """
    Wraps the opencv tracker.
    """

    def __init__(self, vehicle_id):
        self.tracker = new_tracker()
        self.failure_cnt = 0
        self.last_bbox = None
        self.vehicle_id = vehicle_id
        self.lifetime = 0
        self.frame = None
        self.detected = False
        self.category = None

    def plot(self):
        """
        Plot the tracker using matplotlib.
        """
        plt.imshow(self.roi)
        plt.show()

    @property
    def roi(self):
        """
        Get the region of the tracker.
        :return: numpy ndarray
        """
        bbox = self.last_bbox
        img = self.frame
        if bbox is None or img is None:
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


def init_new_tracker(img, bbox, vehicle_id):
    """
    Init new tracker with the region of img specified by the bbox.
    :param vehicle_id: the unique id of the vehicle
    :param img: the whole frame
    :param bbox: the region of the frame
    :return: the initialized brand-new tracker.
    """
    tracker = TrackerWrapper(vehicle_id)
    tracker.init(img, bbox)
    tracker.frame = img
    tracker.detected = True
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
    opts.log_severity_level = 3
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
    num2label = {labels[key]: key for key in labels}

    # Store the former frame's info
    former_frame = None
    former_boxes = []

    # Store current trackers.
    trackers = []

    # Decide when should we drop a tracker.
    tracking_failure_count_upperbound = 5

    # Threshold decide whether the tracker bbox and the detection bbox is the same one.
    IoU_threshold = 0.40

    def update_trackers(img):
        """
        Update all the trackers in the list.
        :param img: the input frame.
        :return: None
        """
        to_be_removed = []
        # Update every tracker.
        for index, tracker in enumerate(trackers):
            # bbox: array of x,y,w,h
            success, bbox = tracker.update(img)
            tracker.lifetime += 1
            tracker.detected = False
            if success:
                tracker.last_bbox = bbox
                tracker.frame = img
            else:
                tracker.last_bbox = None
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

    def draw_detection(img, box, detection_frame=None, transform_bbox=True):
        """
        Draw detection box on img.
        :param img: frame to draw
        :param box: detection bbox
        :return: vehicle id of the detected vehicle
        """
        # Let the transform do the magic!
        if transform_bbox:
            tl, br = box_transform(box, img.shape[0], img.shape[1])
        else:
            tl = (box[0], box[1])
            br = (box[2], box[3])

        # if not len(former_boxes):
        #     ret = generate_car_id()
        # else:
        #     dis_and_id = [(np.linalg.norm(box - former_box), former_id) for former_box, former_id in former_boxes]
        #     min_dis_and_id = min(dis_and_id, key=lambda x: x[0])
        #     if min_dis_and_id[0] > 0.3:
        #         print(f"Tracking failed at {min_dis_and_id[0]}")
        #         ret = generate_car_id()
        #     else:
        #         ret = min_dis_and_id[1]
        # color = colors[ret % len(colors)]
        # color = (255, 128, 0)

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
        # max_iou = -1
        # max_iou_id = None
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
            # Mark the max IoU tracker as detected.
            max_iou_tracker.detected = True
            # Replace max IoU tracker's id with the min id
            max_iou_tracker.vehicle_id = min_id_tracker.vehicle_id
            ret_tracker = max_iou_tracker
            # Remove trackers with smaller IoU
            trackers_to_be_removed = [trackers[x] for x in indices if x != max_iou_index]
            for tracker in trackers_to_be_removed:
                print(f"Remove tracker {tracker.vehicle_id} due to small iou!")
                trackers.remove(tracker)
            vehicle_id = min_id_tracker.vehicle_id  # Store the result vehicle id
        else:
            # We didn't find any matched tracker bbox.
            # It's probably a new vehicle!
            print(f"No iou > IoU_threshold!")
            if br[1] > 900 or br[1] < 100:
                # When the vehicle is at bottom or top, do not track it.
                # If we do track it, may issues do rise!
                print(f"Currently not tracking {tl},{br} since it's at bottom or top")
                # The -1 id means the vehicle is detected but not currently tracked
                vehicle_id = -1
                ret_tracker = None
            else:
                # We find a new car! Give it an id!
                vehicle_id = generate_car_id()
                # Tracking it by adding a new tracker!
                ret_tracker = init_new_tracker(detection_frame, tlbr2xywh(tl, br), vehicle_id)
                trackers.append(ret_tracker)
                print(f"Detected new vehicle! id:{vehicle_id}")
        # Make it colorful and draw it!
        cv2.rectangle(img, tl, br, colors[vehicle_id % len(colors)], 2)
        # Draw the vehicle id!
        cv2.putText(img, str(vehicle_id), (br[0], tl[1] - 10), font, 1.2, (255, 0, 0), 1, cv2.LINE_AA)
        return ret_tracker

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
            cv2.putText(img, str(tracker.vehicle_id), (bbox[0], bbox[1] - 10), font, 1.2, color, 1,
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
            # if category == "sedan":
            #     # if the category is sedan, we should think twice!
            #     largest = np.max(probs)
            #     # Find the second probable choice and compare it with sedan!
            #     second_largest = -1
            #     second_largest_id = -1
            #     for id in range(len(probs)):
            #         value = probs[0, id]
            #         if largest > value > second_largest:
            #             second_largest_id = id
            #             second_largest = value
            #     if largest - second_largest < 0.25:
            #         # if the difference is small, we take the second probable one instead!
            #         category = num2label[second_largest_id]
            if tracker is not None:
                tracker.category = category
        else:
            category = tracker.category
        # Draw the category on the video.
        cv2.putText(img, category, (tl[0], tl[1] - 10), font, 1.2, (255, 33, 100), 1, cv2.LINE_AA)

    def draw_license_plate(img, detection_frame, box):
        tl, br = box_transform(box, detection_frame.shape[0], detection_frame.shape[1])
        vehicle_region = detection_frame[tl[1]:br[1], tl[0]:br[0]]
        license_plate = license_recognition(vehicle_region)
        if license_plate is not None:
            cv2.putText(img, license_plate, (tl[0], tl[1] + 10), font, 1.2, (255, 33, 100), 1, cv2.LINE_AA)

    def process(detection_frame):
        nonlocal former_frame, former_boxes
        # transform the frame for detection.
        transformed = preprocess_image(detection_frame)
        # Copy the frame to draw on it. (Do not mix up the draw frame and the detection frame.)
        draw_frame = detection_frame.copy()
        # Get the results from model.
        boxes, classes, confs = car_detection_sess.run(None, {inp_car_detection: transformed})
        normalized_boxes = (box / 800 for box in boxes)  # Normalize the bboxes
        infos = zip(normalized_boxes, confs)  # Zip normalized boxes and their confidence values.
        # new_former_boxes = []
        update_trackers(detection_frame)  # Update all the trackers.
        # out_bound_trackers = []
        # for tracker in trackers:
        #     if tracker.last_bbox is not None:
        #         for num in tracker.last_bbox:
        #             if num < 0 and tracker.lifetime > 60:
        #                 out_bound_trackers.append(tracker)
        #                 break
        # for tracker in out_bound_trackers:
        #     trackers.remove(tracker)
        #     print(f"Remove {tracker} because it's out of bound.")
        useless_trackers = [tracker for tracker in trackers  # collect useless trackers
                            if tracker.roi is not None and
                            # We think a tracker useless when its region is almost of the same color.
                            # The 20 threshold is just a value from my experience.
                            np.std(tracker.roi) < 32]
        useless_trackers.extend(tracker for tracker in trackers if
                                tracker.last_bbox is not None and
                                # Area too small.
                                tracker.last_bbox[2] * tracker.last_bbox[3] < 1000)
        for tracker in useless_trackers:
            trackers.remove(tracker)
            print(f"Remove {tracker} because it's std or area is too small.")
        for info in infos:
            if info[1] < confidence_lowerbound:
                # We do not draw the detection which is below our confidence lower bound.
                # Just break it, because the array is sorted!
                break
            # car_id = draw_detection(draw_frame, info[0])
            tracker = draw_detection(draw_frame, info[0], detection_frame)
            # new_former_boxes.append((info[0], car_id))
            draw_classification(draw_frame, detection_frame, info[0], tracker)
            if options.license_recognition:
                draw_license_plate(draw_frame, detection_frame, info[0])
        # former_frame = detection_frame
        # former_boxes = new_former_boxes
        to_be_removed = []
        for tracker in trackers:
            if tracker.last_bbox is not None and not tracker.detected:
                print(f"Undetected tracker: {tracker}!")
                roi = tracker.roi
                std = np.std(roi)
                yl = roi.shape[0]
                lower_area = roi[yl // 2:yl, :, :]
                if std < 35:
                    to_be_removed.append(tracker)
                elif std > 45 and np.std(lower_area) > 40:
                    # Recover the detection box
                    tracker.plot()
                    print(f"std: {std}, lower_area |> std: {np.std(lower_area)}.")
                    bbox = tracker.last_bbox
                    bbox = (0 if bbox[0] < 0 else bbox[0], 0 if bbox[1] < 0 else bbox[1], bbox[2], bbox[3])
                    x1y1x2y2_bbox = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
                    draw_detection(draw_frame, x1y1x2y2_bbox,
                                   detection_frame, transform_bbox=False)
                    draw_classification(draw_frame, detection_frame, x1y1x2y2_bbox, tracker, transform_bbox=False)
                    # tracker.plot()
        for tracker in to_be_removed:
            trackers.remove(tracker)
        for tracker in trackers:
            draw_tracker(draw_frame, tracker)  # Visualize all the good trackers。
        return draw_frame

    return process


if __name__ == "__main__":
    # Parser for cmdline arguments
    (options, args) = parser.parse_args()
    if not options.input:
        parser.error("Missing input file!")
    # Init video
    cap = cv2.VideoCapture(options.input)
    result_file = options.output
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    video_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(result_file, video_fourcc, fps_video, (frame_width, frame_height))
    # Init video processing session
    process = init_session()
    # Process video
    frame_id = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Finished!")
            break
        draw_frame = process(frame)
        video_writer.write(draw_frame)
        # Uncomment those if you want the video show.
        # cv2.imshow("Video", draw_frame)
        # cv2.waitKey(20)
        frame_id += 1
    video_writer.release()
    print(f"Processed {frame_id + 1} frames in total!")

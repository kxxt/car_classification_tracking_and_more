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

        def plot(self):
            plt.imshow(self.roi)
            plt.show()

        @property
        def roi(self):
            bbox = self.last_bbox
            img = self.frame
            if bbox is None or img is None:
                return None
            top = bbox[1] if bbox[1] > 0 else 0
            left = bbox[0] if bbox[0] > 0 else 0
            return img[top:top + bbox[3], left:left + bbox[2]]

        def __getattr__(self, item):
            return self.tracker.__getattribute__(item)

        def __repr__(self):
            return f"<Tracker {self.vehicle_id}, bbox={self.last_bbox}, failure_cnt={self.failure_cnt}, lifetime={self.lifetime}>"

    trackers = []

    # Decide when should we drop a tracker.
    tracking_failure_count_upperbound = 5

    IoU_threshold = 0.4

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
        plt.imshow(img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]])
        plt.show()
        return tracker

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
            if success:
                tracker.last_bbox = bbox
                tracker.frame = img
            else:
                tracker.last_bbox = None
                tracker.frame = None
                tracker.failure_cnt += 1
                if tracker.failure_cnt > tracking_failure_count_upperbound:
                    to_be_removed.append(tracker)

        # Remove the useless trackers.
        for tracker in to_be_removed:
            print(f"Remove tracker {tracker.vehicle_id} due to too many failures.")
            trackers.remove(tracker)
        print(f"There are {len(trackers)} trackers now.")

    def preprocess_image(img):
        """
        Preprocess image for vehicle detection.
        :param img:
        :return: preprocess image
        """
        return np.transpose(np.array(cv2.resize(img, (800, 800)), dtype="float32"), (2, 0, 1))[np.newaxis, :] / 255

    def preprocess_vehicle_region(img):
        """
        Preprocess vehicle region for vehicle classification.
        :param img: vehicle region
        :return: preprocess image
        """
        return np.transpose(np.array(cv2.resize(img, (224, 224)), dtype="float32"), (2, 0, 1))[np.newaxis, :] / 255

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
        counter += 1
        return counter

    def draw_detection(img, box, detection_frame=None):
        """
        Draw detection box on img.
        :param img: frame to draw
        :param box: detection bbox
        :return: vehicle id of the detected vehicle
        """
        tl, br = box_transform(box, img.shape[0], img.shape[1])

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
        color = (255, 128, 0)
        cv2.rectangle(img, tl, br, color, 2)
        ious = []
        for index, tracker in enumerate(trackers):
            if tracker.last_bbox is not None:
                bbox = tracker.last_bbox
                ious.append(
                    (
                        index,
                        get_iou(
                            (tl[0], tl[1], br[0], br[1]),
                            (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
                        )
                    )
                )
        # max_iou = -1
        # max_iou_id = None
        indices = []
        for index, iou in ious:
            if iou > IoU_threshold:
                indices.append(index)
        if indices:
            min_id_index = min(indices, key=lambda x: trackers[x].vehicle_id)
            min_id_tracker = trackers[min_id_index]
            max_iou_index = ious[max(range(len(ious)), key=lambda x: ious[x][1])][0]
            max_iou_tracker = trackers[max_iou_index]
            max_iou_tracker.vehicle_id = min_id_tracker.vehicle_id
            trackers_to_be_removed = [trackers[x] for x in indices if x != max_iou_index]
            for tracker in trackers_to_be_removed:
                print(f"Remove tracker {tracker.vehicle_id} due to small iou!")
                trackers.remove(tracker)
            vehicle_id = min_id_tracker.vehicle_id
        else:
            print(f"No iou > IoU_threshold!")
            if br[1] > 900 or br[1] < 100:
                print(f"Currently not tracking {tl},{br} since it's at bottom")
                vehicle_id = -1
            else:
                vehicle_id = generate_car_id()
                trackers.append(init_new_tracker(detection_frame, tlbr2xywh(tl, br), vehicle_id))
                print(f"Detected new vehicle! id:{vehicle_id}")
        cv2.putText(img, str(vehicle_id), (br[0], tl[1] - 10), font, 1.2, (255, 0, 0), 1, cv2.LINE_AA)
        return vehicle_id

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
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 255, 255), 2)
        if tracker.vehicle_id is not None:
            cv2.putText(img, str(tracker.vehicle_id), (bbox[0], bbox[1] - 10), font, 1.2, (255, 255, 255), 1,
                        cv2.LINE_AA)

    def draw_classification(img, detection_frame, box):
        """
        Using detection_frame and box to classify a vehicle and draw the classification result on img
        :param img: frame to draw on
        :param detection_frame: frame that contains the vehicle
        :param box: bbox of the vehicle
        :return: None
        """
        tl, br = box_transform(box, detection_frame.shape[0], detection_frame.shape[1])
        vehicle_region = detection_frame[tl[1]:br[1], tl[0]:br[0]]
        transformed = preprocess_vehicle_region(vehicle_region)
        probs = car_classification_sess.run(None, {inp_car_classification: transformed})[0]
        max_prob_id = np.argmax(probs)
        category = num2label[max_prob_id]
        if category == "sedan":
            largest = np.max(probs)
            second_largest = -1
            second_largest_id = -1
            for id in range(len(probs)):
                value = probs[0, id]
                if largest > value > second_largest:
                    second_largest_id = id
                    second_largest = value
            if largest - second_largest < 0.25:
                category = num2label[second_largest_id]
        cv2.putText(img, category, (tl[0], tl[1] - 10), font, 1.2, (255, 33, 100), 1, cv2.LINE_AA)

    def draw_license_plate(img, detection_frame, box):
        tl, br = box_transform(box, detection_frame.shape[0], detection_frame.shape[1])
        vehicle_region = detection_frame[tl[1]:br[1], tl[0]:br[0]]
        license_plate = license_recognition(vehicle_region)
        if license_plate is not None:
            cv2.putText(img, license_plate, (tl[0], tl[1] + 10), font, 1.2, (255, 33, 100), 1, cv2.LINE_AA)

    def process(detection_frame):
        nonlocal former_frame, former_boxes
        transformed = preprocess_image(detection_frame)
        draw_frame = detection_frame.copy()
        boxes, classes, confs = car_detection_sess.run(None, {inp_car_detection: transformed})
        normalized_boxes = (box / 800 for box in boxes)
        infos = zip(normalized_boxes, confs)
        # new_former_boxes = []
        update_trackers(detection_frame)
        for tracker in trackers:
            draw_tracker(draw_frame, tracker)
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
        useless_trackers = [tracker for tracker in trackers
                            if tracker.roi is not None and
                            np.std(tracker.roi) < 20]
        for tracker in useless_trackers:
            trackers.remove(tracker)
            print(f"Remove {tracker} because it's std is too small.")
        for info in infos:
            if info[1] < confidence_lowerbound:
                break
            # car_id = draw_detection(draw_frame, info[0])
            draw_detection(draw_frame, info[0], detection_frame)
            # new_former_boxes.append((info[0], car_id))
            draw_classification(draw_frame, detection_frame, info[0])
            if options.license_recognition:
                draw_license_plate(draw_frame, detection_frame, info[0])
        # former_frame = detection_frame
        # former_boxes = new_former_boxes
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
    # Init inference session
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
        # cv2.imshow("Video", draw_frame)
        # cv2.waitKey(20)
        frame_id += 1
    video_writer.release()
    print(f"Processed {frame_id + 1} frames in total!")

import json
from math import tan

INPUT_PATH = "velocity_detection/result.json"
OUTPUT_PATH = "velocity_detection/velocities.json"
# decide the span of frame calculate
DETECT_FRAME_SPAN = 5

PARAMETERS = (700.221, -0.08477, -0.191857, 814.976)


def pixel_coordinate_transform(y, param):
    """
    Transform the pixel coordinate to the real distance.
    :param y: the pixel coordinate y.
    :param param: the list of 4 paramaters [A, B, C, D] of transform
    """
    A, B, C, D = param
    return (tan((y - D) / A) - C) / B


def get_velocities(path_or_data=INPUT_PATH, fps=30, params=PARAMETERS):
    # Data process
    if isinstance(path_or_data, str):
        with open(INPUT_PATH) as pre_data:
            data = json.load(pre_data)["frames"]
    else:
        data = path_or_data

    frame_number = len(data)
    # print(frame_number)

    data1 = []
    vehicle_ids = set()
    for frame in data:
        if frame["detections"]:
            for vehicle in frame["detections"]:
                if vehicle["tracker"] is not None:
                    if vehicle["tracker"]["license"] is not None:
                        # (x, y, w, h), (x, y) is the coordinate of top-left
                        pre_pos = vehicle["tracker"]["last_license_bbox"]
                        # y+(h/2)
                        pos = pre_pos[1] + pre_pos[3] / 2
                    else:
                        # y-midpoint of top and bottom
                        pos = (vehicle["tl"][0] + vehicle["br"][0]) / 2
                    data1.append({
                        "frame": frame["frame_id"],
                        "vehicle_id": vehicle["vehicle_id"],
                        "pos": pos,
                        "bbox": {"tl": vehicle["tl"], "br": vehicle["br"]}
                    })
                vehicle_ids.add(vehicle["vehicle_id"])

    # Remove the -1, which stands for undetected car.
    vehicle_ids -= {-1}

    # print(data1)
    # print(vehicle_ids)

    vehicle_ids = list(vehicle_ids)
    vehicle_pos_infos = []

    for vehicle_id in vehicle_ids:
        vehicle_pos_info = []
        for vehicle in data1:
            if vehicle["vehicle_id"] == vehicle_id:
                # bottom_y, top_y
                vehicle_pos_info.append(
                    [
                        vehicle["vehicle_id"],
                        vehicle["pos"],
                        vehicle["frame"],
                        vehicle["bbox"]
                    ]
                )
        vehicle_pos_infos.append(vehicle_pos_info)

    # print(vehicle_pos_infos[0])

    vehicle_velocity_infos = []
    for vehicle_id_index, vehicle in enumerate(vehicle_pos_infos):
        vehicle_id = vehicle_ids[vehicle_id_index]
        # The total pictures of this vehicle
        photo_numbers = len(vehicle)
        # Formula: v = (y2-y1)/(t2-t1)

        # "index" is the index of the list "vehicle_pos_infos"
        # index_1 is the index corresponding to y1 and t1.
        index_1 = 0
        # frame_1 is t1 in the formula above (unit: 1/fps second)
        frame_1 = vehicle[index_1][2]
        for i in range(0, photo_numbers):
            if vehicle[i][2] - frame_1 >= DETECT_FRAME_SPAN - 1:
                # index_2 is the index corresponding to y2 and t2.
                index_2 = i
                # frame_2 is t2
                frame_2 = vehicle[index_2][2]
                # real_pos is y1 and y2 (default unit: meter)
                real_pos_1 = pixel_coordinate_transform(vehicle[index_1][1], params)
                real_pos_2 = pixel_coordinate_transform(vehicle[index_2][1], params)
                # Pay attention to unit change: m*fps -> km/h
                velocity = int(
                    (real_pos_2 - real_pos_1) / (frame_2 - frame_1)
                    * fps * 3.6  # parameters to change unit
                )
                # record the velocity
                for j in range(index_1, index_2 + 1):
                    vehicle[j].append(velocity)

                index_1 = index_2 + 1
            # If we haven't reached the end:
            if index_1 != photo_numbers:
                frame_1 = vehicle[index_1][2]

        # If there are still frames that we did't calculate the velocity:
        if index_1 != photo_numbers:
            for j in range(index_1, photo_numbers):
                vehicle[j].append(velocity)

        vehicle_velocity_infos.append(vehicle)

    # print(vehicle_velocity_infos[5])
    # print(len(vehicle_velocity_infos) - len(vehicle_pos_infos))

    velocity_by_frame = []
    for frame_id in range(frame_number):
        frame_vehicles = []
        for vehicle_list in vehicle_velocity_infos:
            for vehicle in vehicle_list:
                if vehicle[2] == frame_id:
                    frame_vehicles.append(
                        {
                            "vehicle_id": vehicle[0],
                            "velocity": vehicle[4],
                            "bbox": vehicle[3]
                        }
                    )
        velocity_by_frame.append(frame_vehicles)
    return velocity_by_frame


if __name__ == '__main__':
    velocity_by_frame = get_velocities()

    with open(OUTPUT_PATH, mode='w') as velocity_file:
        json.dump(velocity_by_frame, velocity_file)

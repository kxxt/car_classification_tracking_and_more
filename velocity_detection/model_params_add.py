import json
import pandas as pd


PATH = "velocity_detection/result.json"
SEDAN_LENGTH = 5
START_PIXEL_Y = 680

def read_data(input_file):
    with open(input_file) as pre_data:
        pre_data = pre_data.read()
        data = json.loads(pre_data)["frames"]
    return data

# pick out the sedans in the data
def pick_out_sedans(data):
    data1 = []
    vehicle_ids = set()
    for frame in data:
        if frame["detections"]:
            for vehicle in frame["detections"]:
                if vehicle["tracker"] is not None and \
                    vehicle["tracker"]["category"]=="sedan":
                    data1.append(vehicle)
                    vehicle_ids.add(vehicle["vehicle_id"])
    vehicle_ids = list(vehicle_ids)
    return vehicle_ids, data1


def sedans_by_ids(vehicle_ids, data1):
    sedan_pos_infos = []
    for vehicle_id in vehicle_ids:
        vehicle_pos_info = []
        for sedan in data1:
            if sedan["vehicle_id"] == vehicle_id:
                # bottom_y, top_y
                vehicle_pos_info.append((sedan["br"][1], sedan["tl"][1]))
        sedan_pos_infos.append(vehicle_pos_info)
    return sedan_pos_infos

data = read_data(PATH)
vehicle_ids, sedans = pick_out_sedans(data)
del data
sedan_pos_infos = sedans_by_ids(vehicle_ids, sedans)


training_data = []
for index, sedan_pos in enumerate(sedan_pos_infos):
    # sedan_pos: a list of [bottom_y, top_y]
    if sedan_pos[0][0] < START_PIXEL_Y:
        sedan_pos_infos.pop(index)
        vehicle_ids.pop(index)

for sedan_pos in sedan_pos_infos:
    pos_record = []
    compare_value = START_PIXEL_Y
    for i in range(0, len(sedan_pos)):
        # If i is the last: break it, preventing index i+1 raising error.
        if i == len(sedan_pos) - 1 :
            break
        print(sedan_pos[i][0])
        if sedan_pos[i][0] >= compare_value and sedan_pos[i+1][0] < compare_value:
            if sedan_pos[i+1][0] - compare_value < compare_value - sedan_pos[i][0]:
                # Record the related top_y of the i-th position.
                compare_value = sedan_pos[i][1]
                # Record the position of bottom
                pos_record.append(sedan_pos[i][0])
            else:
                compare_value = sedan_pos[i+1][1]
                pos_record.append(sedan_pos[i+1][0])
    if len(pos_record) >= 4:
        # Suppose each sedan has the same length, and add the data
        for index, pixel_pos in enumerate(pos_record):
            training_data.append([index * SEDAN_LENGTH, pixel_pos])


# print(training_data)
training_list = json.dumps(training_data)
with open("velocity_detection/training_data.json", mode='w') as training_file:
    training_file.write(training_list)

training_data = pd.DataFrame(training_data)

training_data.to_csv("velocity_detection/training_data.csv")

# data1_json = json.dumps(data1)
# with open("velocity_detection/regression.json", mode="w") as regression_data:
#     regression_data.write(data1_json)

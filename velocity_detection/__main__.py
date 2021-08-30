from sys import argv
import json
import cv2
from . import generate_training_data, find_fit, get_velocities

if len(argv) < 3:
    print("""Error: Expected at least 2 arguments! 
the first arugment is the video's analysis result json file,
the second argument is the input video path,
the third argument (optional, default to "result.mp4") is the output video path.
Aborted!
""")
    exit(-1)

analysis_file = argv[1]
input_file = argv[2]
output_file = argv[3] if len(argv) > 3 else "result.mp4"

with open(analysis_file) as f:
    data = json.load(f)
fps = int(data['fps'])
frames = data['frames']
training_data = generate_training_data(frames)
fitted = find_fit(training_data)
print(f"Found fitted parameters: {fitted}")
velocities = get_velocities(frames, fps, fitted)

cap = cv2.VideoCapture(input_file)
fps_video = cap.get(cv2.CAP_PROP_FPS)
video_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_writer = cv2.VideoWriter(output_file, video_fourcc, fps_video, (frame_width, frame_height))

frame_id = 0  # Store the frame id

viter = iter(velocities)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        # We reached the end of the video.
        print("Finished!")
        break
    analyses = next(viter)
    for analysis in analyses:
        if analysis['velocity'] is not None:
            cv2.putText(frame, str(analysis['velocity']) + " km/h",
                        (analysis['bbox']["br"][0] + 5, analysis["bbox"]["tl"][1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2,
                        cv2.LINE_AA)
    video_writer.write(frame)
    frame_id += 1  # Increase the frame id counter
video_writer.release()  # Release the video file since we've finished processing.

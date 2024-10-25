import cv2
from ultralytics import YOLO


model = YOLO("yolo11n-pose.pt")


# Open the video file
video_path = "Person.mp4"
cap = cv2.VideoCapture(video_path)

# Get the video's width, height, and frames per second (fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 'mp4v' is the codec for .mp4
output_path = "PersonPose.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model(frame)

        # Check if any detections are made
        if results and len(results) > 0:
            # Extract the first detection's keypoints
            keypoints = results[0].keypoints

            # Ensure keypoints exist and are valid
            if keypoints is not None and keypoints.xy is not None:
                xy = keypoints.xy
                keypoint_set = xy[0]  # shape: (K, 2) where K is the number of keypoints

                # Only proceed if the keypoint_set is not empty
                if len(keypoint_set) != 0:  # Ensure there are at least 17 keypoints for a full pose
                    # Extract keypoints with their indices for specific body parts
                    nose = keypoint_set[0]
                    left_eye = keypoint_set[1]
                    right_eye = keypoint_set[2]
                    left_ear = keypoint_set[3]
                    right_ear = keypoint_set[4]
                    left_shoulder = keypoint_set[5]
                    right_shoulder = keypoint_set[6]
                    left_elbow = keypoint_set[7]
                    right_elbow = keypoint_set[8]
                    left_wrist = keypoint_set[9]
                    right_wrist = keypoint_set[10]
                    left_hip = keypoint_set[11]
                    right_hip = keypoint_set[12]
                    left_knee = keypoint_set[13]
                    right_knee = keypoint_set[14]
                    left_ankle = keypoint_set[15]
                    right_ankle = keypoint_set[16]

                    # Example of drawing a line between left_eye and left_ankle

                    cv2.line(frame, tuple(map(int, left_eye)), tuple(map(int, right_eye)), color=(255,0, 0), thickness=2)
                    cv2.line(frame, tuple(map(int, left_shoulder)), tuple(map(int, right_shoulder)), color=(255,0, 0), thickness=2)
                    cv2.line(frame, tuple(map(int, left_shoulder)), tuple(map(int, left_elbow)), color=(255,0, 0), thickness=2)
                    cv2.line(frame, tuple(map(int, left_elbow)), tuple(map(int, left_wrist)), color=(255,0, 0), thickness=2)
                    cv2.line(frame, tuple(map(int, right_elbow)), tuple(map(int, right_shoulder)), color=(255,0, 0), thickness=2)
                    cv2.line(frame, tuple(map(int, right_elbow)), tuple(map(int, right_wrist)), color=(255,0, 0), thickness=2)
                    cv2.line(frame, tuple(map(int, right_hip)), tuple(map(int, left_hip)), color=(255,0, 0), thickness=2)
                    cv2.line(frame, tuple(map(int, right_hip)), tuple(map(int, right_knee)), color=(255,0, 0), thickness=2)
                    cv2.line(frame, tuple(map(int, right_knee)), tuple(map(int, right_ankle)), color=(255,0, 0), thickness=2)
                    cv2.line(frame, tuple(map(int, left_hip)), tuple(map(int, left_knee)), color=(255,0, 0), thickness=2)
                    cv2.line(frame, tuple(map(int, left_knee)), tuple(map(int, left_ankle)), color=(255,0, 0), thickness=2)

        # Write the processed frame to the output video file
        out.write(frame)

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video saved successfully!")

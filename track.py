from ultralytics import YOLO
import time
import cv2
from utils import pose_track

model = YOLO("yolov8n-pose.pt")
video_path = "./videos/indoor1.MOV"

# results = model.track(source="./videos/indoor1.MOV", show=True) 

frame_count = 0
frame_save_interval = 5


# Run through the video and infer on each frame
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_count % frame_save_interval == 0:
        results = model.track(frame)[0]
    
        # Print the bounding boxes

        person_obj = results.boxes
        person_ids = person_obj.id
        print(person_ids)
        person_cls = person_obj.cls
        person_confs = person_obj.conf        
        person_bbox = person_obj.xywhn
        
        person_keypoints = results.keypoints
        person_keypoints_conf =  person_keypoints.conf
        person_keypoints_coord = person_keypoints.xyn
        
        drawn_frame = pose_track(frame, person_cls, person_confs, person_ids, person_bbox, person_keypoints_coord, draw=True, save=False, save_path="./")

        # Show the image
        cv2.imshow('frame', drawn_frame)
        # cv2.imwrite(f"track_1_{frame_count}.jpg", drawn_frame)
        # time.sleep(5)
        
        
    
    frame_count += 1
    # if frame_count == 2:
    #     break
    
    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)


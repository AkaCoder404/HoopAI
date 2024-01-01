import cv2
import os
import subprocess
import numpy as np

def pose():
    """
    Handles keypoint detection only
    """

def pose_track(frame, clss, confs, ids, boxes, keypoints, draw=True, save=False, save_path="./"):
    """
    Handles keypoint detection with object tracking
    
    param: frame: frame to be processed
    param: draw: whether to draw the bounding boxes and keypoints
    param: save: whether the keypoint detection results in coco8 format
    """
    
    # 5 Different colors
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255)]
    frame_height, frame_width, _ = frame.shape
    
    for i in range(len(clss)):
        c = clss[i]
        conf = confs[i]
        id = ids[i]
        box = boxes[i]
        keypoint = keypoints[i]
        
        # Draw the bounding box and label by id
        if draw:
            color = colors[int(id)]
            x, y, w, h = box
            
            # Normalize the keypoints
            x, y, w, h = int(x*frame_width), int(y*frame_height), int(w*frame_width), int(h*frame_height)
            
            cv2.putText(frame, f"{id}", (int(x - w/2), int(y - h/2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), color, 2)
         
    return frame

def caculate_angle():
    """
    Caculate the angle between two keypoints
    """
    pass

def caculate_distance():
    """
    Caculate the distance between two keypoints
    """
    pass

def draw_person_ball(frame, person, person_kp, ball, save=False, save_path="./"):
    """
    Draw the person and ball on the frame
    """
    person_cls = person.cls
    person_confs = person.conf
    person_bbox = person.xywhn
    
    person_keypoints = person_kp
    person_keypoints_conf =  person_keypoints.conf
    person_keypoints_coord = person_keypoints.xyn
    
    
    ball_cls = ball.cls
    ball_confs = ball.conf
    ball_bbox = ball.xywhn
    
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255)]
    frame_height, frame_width, _ = frame.shape
    
    if save:
        f = open(save_path, "w")
    
    # Draw person
    for i in range(len(person_cls)):
        if person_confs[i] < 0.75:
            continue
        
        c = int(person_cls[i])
        # Draw the bounding box
        x, y, w, h = person_bbox[i][0], person_bbox[i][1], person_bbox[i][2], person_bbox[i][3]
        dx, dy, dw, dh = int(x*frame_width), int(y*frame_height), int(w*frame_width), int(h*frame_height)
        cv2.rectangle(frame, (int(dx - dw/2), int(dy - dh/2)), (int(dx + dw/2), int(dy + dh/2)), colors[c], 2)
        
        if save:
            f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f} ")    
                
        # Draw keypoints
        for j in range(len(person_keypoints_coord[i])):
            kx, ky = person_keypoints_coord[i][j][0], person_keypoints_coord[i][j][1]
            dkx, dky = int(kx*frame_width), int(ky*frame_height)
            
            if kx == 0 and ky == 0: # Keypoint not detected
                if save:
                    f.write(f"{kx:.6f} {ky:.6f} {0.0:.6f} ")
                continue
            
            if save:
                f.write(f"{kx:.6f} {ky:.6f} {2.0:.6f} ")
            
            cv2.circle(frame, (dkx, dky), 3, colors[c], -1) 
        
    # Draw ball bounding box and keypoints
    for i in range(len(ball_cls)):
        if ball_confs[i] < 0.50:
            continue
        
        c = int(ball_cls[i]) + 1
        
        # Draw bounding box
        x, y, w, h = ball_bbox[i][0], ball_bbox[i][1], ball_bbox[i][2], ball_bbox[i][3]
        dx, dy, dw, dh = int(x*frame_width), int(y*frame_height), int(w*frame_width), int(h*frame_height)
        cv2.rectangle(frame, (int(dx - dw/2), int(dy - dh/2)), (int(dx + dw/2), int(dy + dh/2)), colors[c], 2)
        
        if save:
            f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f} ")
        
        # Draw keypoints
        if c == 2 or c == 3: # Ball
            cv2.circle(frame, (int(x), int(y)), 5, colors[c], -1)
        # else: # Rim
        #     cv2.circle(frame, (int(x), int(y + h/2)), 3, colors[c], -1)
        
        if save:
            f.write(f"{x:.6f} {y:.6f} {2.0:.6f} ")

            # The next 16 keypoints are all 0
            for j in range(16):
                f.write(f"{0.0:.6f} {0.0:.6f} {0.0:.6f} ")
            
        

            
    return frame

def draw():
    """
    Draw the bounding boxes and keypoints on the frame
    """
  
def convert_normalized_to_standard(center_x, center_y, width, height):
    """
    Convert normalized bounding box coordinates (center_x, center_y, width, height)
    to standard coordinates (top-left x, top-left y, bottom-right x, bottom-right y).
    """
    half_width, half_height = width / 2, height / 2
    top_left_x = center_x - half_width
    top_left_y = center_y - half_height
    bottom_right_x = center_x + half_width
    bottom_right_y = center_y + half_height
    return top_left_x, top_left_y, bottom_right_x, bottom_right_y
   
def is_within_box(box1, box2):
    """
    Check if bounding box1 is completely within bounding box2.
    The boxes are given in the format (center_x, center_y, width, height).
    """
    # Convert normalized coordinates to standard coordinates
    box1 = convert_normalized_to_standard(*box1)
    box2 = convert_normalized_to_standard(*box2)

    # Check if box1 is within box2
    return (box1[0] >= box2[0] and box1[1] >= box2[1] and
            box1[2] <= box2[2] and box1[3] <= box2[3])

def denormalize_box(box, frame):
    """
    Denormalize the bounding box given the frame
    """
    frame_height, frame_width, _ = frame.shape
    x, y, w, h = box[0] * frame_width, box[1] * frame_height, box[2] * frame_width, box[3] * frame_height
    return [x, y, w, h]

def denormalize_kp(kp_pair, frame):
    """
    Denormalize the keypoint pair given the frame
    """
    frame_height, frame_width, _ = frame.shape
    x, y = kp_pair[0] * frame_width, kp_pair[1] * frame_height
    return [x, y]


def detect_score(rim, ball, frame, confidence_threshold=0.50):
    ball_cls = ball.cls
    ball_confs = ball.conf
    ball_boxes = ball.xywhn
    
    # Iterate through ball_cls
    for i in range(len(ball_cls)):
        # if ball_confs[i] < 0.50:
        #     continue
    
        bbox = ball_boxes[i]
        
        # Denormalize bounding box
        image_height, image_width, _ = frame.shape
        bbox[0], bbox[1], bbox[2], bbox[3] = bbox[0] * image_width, bbox[1] * image_height, bbox[2] * image_width, bbox[3] * image_height
        
        rim_copy = rim.copy()
        rim_copy[0], rim_copy[1], rim_copy[2], rim_copy[3] = rim_copy[0] * image_width, rim[1] * image_height, rim[2] * image_width, rim[3] * image_height
        
        if is_within_box(bbox, rim_copy):
            return True
        
    return False

def draw_box(frame, box, color=(0,255,0)):
    """
    Draw a bounding box on the frame given normalized coordinates (center_x, center_y, width, height).
    """
    frame_height, frame_width, _ = frame.shape
    x, y, w, h = box[0] * frame_width, box[1] * frame_height, box[2] * frame_width, box[3] * frame_height 

    # Draw bounding box
    cv2.rectangle(frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), color, 2)
    return frame

def create_gif(input_images_path, output_path):
    """
    Create a gif from a list of images
    """
    # Convert paths to absolute paths
    input_images_path = os.path.abspath(input_images_path)
    output_path = os.path.abspath(output_path)
    fps = 30
    
    # Get a list of image files in the directory
    image_files = sorted([os.path.join(input_images_path, f) for f in os.listdir(input_images_path) if os.path.isfile(os.path.join(input_images_path, f))])
    
    # Create a temporary file and write the file list into it
    file_list = "./tmp/filelist.txt"
    with open(file_list, 'w') as f:
        for image_file in image_files:
            f.write(f"file '{image_file}'\n")
    
    # Run ffmpeg to create the gif
    subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', file_list, '-vf', f"fps={fps}", '-pix_fmt', 'rgb24', output_path])
    
    # Remove the temporary file
    os.remove(file_list)
    
    
def detect_rim(frame, model):
    """
    Detect the rim in the frame and return the bounding box
    """
    pass


def calculate_overlap(bbox1, bbox2):
    """
    Calculate the percentage of overlap of bbox1 with bbox2.
    
    Each bounding box is represented as a tuple (x, y, width, height) where x and y are 
    the coordinates of the top-left corner, and width and height are normalized.

    :param bbox1: Tuple representing the first bounding box.
    :param bbox2: Tuple representing the second bounding box.
    :return: Percentage of bbox1 that overlaps with bbox2.
    """
    cx1, cy1, w1, h1 = bbox1
    cx2, cy2, w2, h2 = bbox2

    # Calculating the half widths and half heights
    half_w1, half_h1 = w1 / 2, h1 / 2
    half_w2, half_h2 = w2 / 2, h2 / 2

    # Calculating the edges of the boxes
    left1, right1 = cx1 - half_w1, cx1 + half_w1
    top1, bottom1 = cy1 + half_h1, cy1 - half_h1

    left2, right2 = cx2 - half_w2, cx2 + half_w2
    top2, bottom2 = cy2 + half_h2, cy2 - half_h2

    # Calculating the intersection area
    intersect_left = max(left1, left2)
    intersect_right = min(right1, right2)
    intersect_top = min(top1, top2)
    intersect_bottom = max(bottom1, bottom2)

    # Check if there is no intersection
    if intersect_right < intersect_left or intersect_top < intersect_bottom:
        return 0.0

    # Area of intersection
    intersect_area = (intersect_right - intersect_left) * (intersect_top - intersect_bottom)

    # Area of the first box
    area_box1 = w1 * h1

    # Percentage of box1 inside box2
    return (intersect_area / area_box1) * 100

def detect_holding_ball(frame, person_obj, person_keypoints, ball):
    """
    Detect if the person is holding the ball, in other words, if the ball has some overlap with the person
    """
    ball_boxes = ball.xywhn
    person_boxes = person_obj.xywhn
    overlap = 0.0
    
    # Iterate through ball_boxes
    for ball in ball_boxes:
        # Iterate through person_boxes
        ball = denormalize_box(ball, frame)
        for person in person_boxes:
            person = denormalize_box(person, frame)
            overlap = calculate_overlap(ball, person)
            print(overlap)
            if overlap > 0.10:
                return True, overlap
            
    return False, overlap
    
    
    
keypoints_map = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle"    
}

keypoints_map_reverse = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16    
}

def calculate_joint_angle(a, b, c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

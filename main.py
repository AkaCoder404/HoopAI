import os
import sys
import cv2
import argparse
from ultralytics import YOLO
from utils import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str, default='./videos/input.mp4', help='path to the input video')
    parser.add_argument('--show_video' , type=bool, default=True)
    
    parser.add_argument('--save_frames', type=bool, default=False)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--output_keypoints', type=bool, default=False)

    parser.add_argument('--image_height', type=int, default=480)
    parser.add_argument('--image_width', type=int, default=640)
    parser.add_argument('--frame_save_interval', type=int, default=5)
    
    # parser.add_argument('--keypoint_detection', type=bool, default=True')
    parser.add_argument('--ball_detection', type=bool, default=True)
    parser.add_argument('--player_model', type=str, default='./models/yolov8s-pose.pt')
    parser.add_argument('--ball_model', type=str, default='./models/yolov8s_bbal_detector_mAP50_95_85.pt')
    parser.add_argument('--rim_model', type=str, default='./models/yolov8s_ball_detector.pt')
    parser.add_argument('--default_rim', type=list, default=[0.8528, 0.3129, 0.0800, 0.0950], help='[x, y, w, h]')
    
    
    parser.add_argument('--show_stats', type=bool, default=False)
    parser.add_argument('--calculate_angles', type=bool, default=False)
    
    parser.add_argument('--create_gif', type=bool, default=False)
    return parser.parse_args()
    

def main(args):
    frame_count = 0
    saved_frame_count = 0
    frame_save_interval = args.frame_save_interval
    input_video = args.input_video
    
    shots_made = 0
    shots_taken = 0
    score_delay = 15
    frame_of_previous_shot_made = 0
    state_holding_ball = False
    score_release_angles = []
    
    # [{"plotted_frame": 0, "plotted_location": (x, y), "plotted_text": "text", "wait_frames": 5"}]
    plot_buffer = [] # Things to plot for a couple of frames before clearing
    
    
    # Make output directory if it doesn't exist
    if args.save_frames:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    
    # Models
    person_model = YOLO(args.player_model)
    ball_model   = YOLO(args.ball_model)
    
    f = open("status.txt", "a")
    
    if args.default_rim == ['N', 'o', 'n', 'e']:
        # TODO Temporary hack
        print("Finding rim...")
        is_found = False
        model = YOLO(args.rim_model)
        cap = cv2.VideoCapture(input_video)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model(frame, imgsz=640, conf=0.5)[0]
            rim_obj = results.boxes
            rim_cls = rim_obj.cls
            rim_confs = rim_obj.conf
            rim_bbox = rim_obj.xywhn
            
            print("rim_cls", rim_cls)
            for i in range(len(rim_cls)):
                if int(rim_cls[i]) == 0:
                    rim_bbox = rim_bbox[i]
                    draw_box(frame, rim_bbox, color=(255,255,255))
                    cv2.imwrite("rim.jpg", frame)
                    is_found = True
                    break
                
            if is_found:
                print("Found rim!")
                args.default_rim = rim_bbox.tolist()
                break
        cv2.destroyAllWindows()
        cap.release()
    
    cap = cv2.VideoCapture(input_video)
    while cap.isOpened():  
        ret, frame = cap.read()
        if not ret:
            break
    
        # Run inference on the frame
        if frame_count % frame_save_interval == 0:
            
            # Resize the frame if needed
            frame = cv2.resize(frame, (args.image_width, args.image_height))
            frame_height, frame_width, _ = frame.shape
            orig_frame = frame.copy()
            
            # Draw the rim, detect the score
            rim = args.default_rim
            drawn_frame = draw_box(frame, rim, color=(255,255,255))
            
            # Perform inference on frame using YOLOv8 keypoint detection
            person_results  = person_model(frame, imgsz=640, conf=0.75)[0]
            person_obj      = person_results.boxes
            # person_ids      = person_obj.id
            person_cls      = person_obj.cls
            person_confs    = person_obj.conf
            person_bbox     = person_obj.xywhn
            
            person_keypoints = person_results.keypoints
            person_keypoints_conf =  person_keypoints.conf
            person_keypoints_coord = person_keypoints.xyn
            
            # Perform inference on frame using YOLOv8 ball detection
            ball_results = ball_model(frame, imgsz=640, conf=0.35)[0]
            ball_obj     = ball_results.boxes
            # ball_ids     = ball_obj.id
            ball_cls     = ball_obj.cls
            ball_confs   = ball_obj.conf
            ball_bbox    = ball_obj.xywhn
            
            # Draw the bounding boxes
            drawn_frame = draw_person_ball(frame, person_obj, person_keypoints, ball_obj, save=args.output_keypoints, save_path=args.output_dir + f"/frame_{frame_count:06d}.txt")
        
            # Detect if holding the ball
            is_holding_ball, overlap_percent = detect_holding_ball(frame, person_obj, person_keypoints, ball_obj)
            if is_holding_ball and not state_holding_ball:
                f.write(f"{frame_count} picked up the ball... {overlap_percent}\n")
            elif not is_holding_ball and state_holding_ball:
                f.write(f"{frame_count} shot the ball... {overlap_percent}\n")
                shots_taken += 1
            else:
                f.write(f"{frame_count} holding the ball... {overlap_percent}\n")

            # Detect release angle
            # TODO Assuming there is only one person in the frame
            person_keypoints_coord = person_keypoints_coord[0]
            if len(person_keypoints_coord) > 0:
                right_shoulder = person_keypoints_coord[keypoints_map_reverse['right_shoulder']]
                right_elbow = person_keypoints_coord[keypoints_map_reverse['right_elbow']]
                right_wrist = person_keypoints_coord[keypoints_map_reverse['right_wrist']]
                
                # Calculate the angles                
                rs = denormalize_kp(right_shoulder, frame)
                re = denormalize_kp(right_elbow, frame)
                rw = denormalize_kp(right_wrist, frame)
                
                        
                # cv2.putText(drawn_frame, f"RS: ({rs[0]:.2f}, {rs[1]:.2f})", (int(rs[0]), int(rs[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                # cv2.putText(drawn_frame, f"RE: ({re[0]:.2f}, {re[1]:.2f})", (int(re[0]), int(re[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                # cv2.putText(drawn_frame, f"RW: ({rw[0]:.2f}, {rw[1]:.2f})", (int(rw[0]), int(rw[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                rs_re_rw_angle = calculate_joint_angle(rs, re, rw)
                
                if not is_holding_ball and state_holding_ball:
                    cv2.putText(drawn_frame, f"Release Angle: {rs_re_rw_angle:.2f}", (frame_width - 375, frame_height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    plot_buffer.append({"plotted_frame": frame_count, "plotted_location": (frame_width - 375, frame_height - 25), "plotted_text": f"Release Angle: {rs_re_rw_angle:.2f}", "wait_frames": 15})
                    f.write(f"{frame_count} release angle {rs_re_rw_angle:.2f}\n")
                
            
            # Detect the score, detect if the ball is in the rim
            is_score = detect_score(rim, ball_obj, frame)
            
            if is_score and frame_count - frame_of_previous_shot_made > score_delay:
                cv2.putText(drawn_frame, "SCORE!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                print(f"Score at frame {frame_count}")
                f.write(f"{frame_count} score {shots_made}\n")
                frame_of_previous_shot_made = frame_count
                shots_made += 1
                score_release_angles.append(rs_re_rw_angle)
                
                plot_buffer.append({"plotted_frame": frame_count, "plotted_location": (50, 50), "plotted_text": "SCORE!", "wait_frames": 15})
                
                
            # Show stats
            if args.show_stats:
                # Draw the stats in bottom right corner
                ## Draw a background rectangle in the bottom right corner
                cv2.rectangle(drawn_frame, (frame_width - 375, frame_height - 125), (frame_width, frame_height), (0, 0, 0), -1)
                cv2.putText(drawn_frame, f"Frame: {frame_count}", (frame_width - 375, frame_height - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(drawn_frame, f"Shots taken: {shots_taken}", (frame_width - 375, frame_height - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(drawn_frame, f"Shots made: {shots_made}", (frame_width - 375, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # TODO Assuming there is only one person in the frame
                # person_keypoints_coord = person_keypoints_coord[0]
                # right_shoulder = person_keypoints_coord[keypoints_map_reverse['right_shoulder']]
                # right_elbow = person_keypoints_coord[keypoints_map_reverse['right_elbow']]
                # right_wrist = person_keypoints_coord[keypoints_map_reverse['right_wrist']]
                
                # # Calculate the angles                
                # rs = denormalize_kp(right_shoulder, frame)
                # re = denormalize_kp(right_elbow, frame)
                # rw = denormalize_kp(right_wrist, frame)
                # rs_re_rw_angle = calculate_joint_angle(rs, re, rw)
                
                # # cv2.putText(drawn_frame, f"Right shoulder: ({rs[0]:.2f}, {rs[1]:.2f})", (int(rs[0]), int(rs[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # # If just shot the ball, show release angle
                # if not is_holding_ball and state_holding_ball:
                #     cv2.putText(drawn_frame, f"Release Angle: {rs_re_rw_angle:.2f}", (frame_width - 400, frame_height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                #     plot_buffer.append({"plotted_frame": frame_count, "plotted_location": (frame_width - 400, frame_height - 25), "plotted_text": f"Release Angle: {rs_re_rw_angle:.2f}", "wait_frames": 15})
                #     f.write(f"{frame_count} release angle {rs_re_rw_angle:.2f}\n")
                
            # Plot the buffer
            for plot in plot_buffer:
                cv2.putText(drawn_frame, plot["plotted_text"], plot["plotted_location"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                plot["wait_frames"] -= 1
                if plot["wait_frames"] == 0:
                    plot_buffer.remove(plot)
                
            # Update the state
            state_holding_ball = is_holding_ball           
                
            # Show the frame
            if args.show_video:
                cv2.imshow('frame', drawn_frame)
                
            # Save the frame
            if args.save_frames:
                cv2.imwrite(os.path.join(args.output_dir, f'frame_{frame_count:06d}.jpg'), drawn_frame)
                saved_frame_count += 1
                      
        frame_count += 1    
            
        # Quit the program if 'q' is pressed
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            print("Quit")
            break

    # Print stats
    print("-" * 50)
    print(f"Total frames: {frame_count}")
    print(f"Saved frames: {saved_frame_count}")
    print(f"Shots taken: {shots_taken}")
    print(f"Shots made: {shots_made}")
    print(f"Score percentage: {shots_made / shots_taken * 100:.2f}%")
    print(f"Average release angle: {sum(score_release_angles) / len(score_release_angles):.2f}")
    print("-" * 50)
    
if __name__ == "__main__":
    args = get_args()
    main(args)
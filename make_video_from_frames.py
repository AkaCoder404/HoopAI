'''
Title: Make Video from Frames

'''

import os
import cv2
import subprocess

input_path = "./output/run1"
output_path = "./videos/v2_output.gif"
fps = 30

# Convert paths to absolute paths
input_path = os.path.abspath(input_path)
output_path = os.path.abspath(output_path)

# Get a list of image files in the directory
image_files = sorted([os.path.join(input_path, f) for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.endswith(".jpg")])

# Create a temporary file and write the file list into it
file_list = "./tmp/filelist.txt"
with open(file_list, 'w') as f:
    for image_file in image_files:
        f.write(f"file '{image_file}'\n")

# Run ffmpeg to create the gif
subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', file_list, '-vf', f"fps={fps}", '-pix_fmt', 'rgb24', output_path])

# Hoopai
Predicting basketball shot probability using computer vision

## Environment
python 3.8.18

### Models and Dataset
The datasets used to train are located:
- Here...
- Here...

The trained models can be found
- Here...
- Here...

### Tools
Used labelmestudio.

### Running
Put video in ...
```
python main.py --input_video './videos/indoor2.mp4' --save_frames True --output_dir ./output/run1 --show_stats True   
```

## Immediate Todos
- [ ] Improve basketball/rim detection for when ball and rim net overlap by improving dataset
- [ ] Combine basketball/rim/pose detection into one.


## Eventual Todos
- [ ] Set up simple LSTM RNN for shot prediction based on keypoints.
- [ ] User frontend built using React to allow users to use.
- [ ] Build mobile application
- [ ] Basketball shot arc prediction: (1) Linear regression (2) Physics?
- [ ] Improved method to handle frames where ball is not detected?

## Development
- 2023.X.X V2: Improved ball and rim detection with bigger dataset.
  - Simple state (holding, shot, score) recognition using overlap of bounding boxes (ball on person, ball on rim).
  - 


- 2023.12.08 V1: Proof of concept with Yolov8 for pose detection, ball detection and Rim Detection
  - Strictly python backend. Run through video frame by frame with CV2. Run inference using trained models.
  - Use two seperate models, one to perform pose detection, one to perform ball and rim detection


![](images/v1_output.gif)
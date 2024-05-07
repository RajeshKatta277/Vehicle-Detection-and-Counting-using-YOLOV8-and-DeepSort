from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import os 
import subprocess
import IPython as ipy
from tqdm.notebook import tqdm

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from deep_sort.sort.tracker import Tracker

%matplotlib inline



class VehicleDetector:
    def __init__(self):
        self.class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.model = YOLO('yolov8x.pt') 
        deepsort_weights='/kaggle/working/Tracking-and-counting-Using-YOLOv8-and-DeepSORT/deep_sort/deep/checkpoint/ckpt.t7'
        self.tracker=DeepSort(model_path=deepsort_weights,max_age=100,max_iou_distance=0.8)
        self.down={}
        self.up={}
        self.counter_down=[]
        self.counter_up=[]
        self.red_line_y=700
        self.blue_line_y=750
        self.offset = 6 
        
    def capture(self,video):
        cap=cv2.VideoCapture(video)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)
        return cap,width,height,fps
    
    def fetch_video_writer(self,name,width,height,fps):
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')  
        self.output_video = cv2.VideoWriter(name, fourcc, fps, (width, height))
    
    def write_video(self,frame):
        self.output_video.write(frame)
    
    def resize_frame(self,frame,dim=None):
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA) 
    
    def get_detections(self,frame):
        results=self.model.predict(frame,classes=[2,3,5,7],device=0,verbose=True)
        for result in results:
            boxes=result.boxes
            probs=result.probs
            cl=boxes.cls.tolist()
            xyxy=boxes.xyxy
            conf=boxes.conf
            xywh=boxes.xywh
        pred_cls=np.array(cl)
        conf=conf.detach().cpu().numpy()
        xyxy=xyxy.detach().cpu().numpy()
        bboxes_xywh=xywh
        bboxes_xywh=xywh.cpu().numpy()
        bboxes_xywh=np.array(bboxes_xywh,dtype='float') 
        for i in range(len(pred_cls)):
            bbox = xyxy[i]
            class_id = int(pred_cls[i])
            confidence = conf[i]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 3)  
            label = f'{self.class_list[class_id]}: {confidence:.2f}'
            cv2.putText(frame, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        return bboxes_xywh,conf,frame
    
    def count(self,bboxes_xywh,conf,frame):
        tracks=self.tracker.update(bboxes_xywh,conf,frame) 
    
        for track in self.tracker.tracker.tracks:
            track_id=track.track_id
            hits=track.hits
            x3,y3,x4,y4=track.to_tlbr()
            cx=int((x3+x4)//2)
            cy=int((y3+y4)//2)
            
            if track_id not in self.down:
                if self.red_line_y < (cy + self.offset) and self.red_line_y > (cy - self.offset):
                    self.down[track_id]=cy   
            if track_id in self.down :
                if self.blue_line_y < (cy + self.offset) and self.blue_line_y > (cy - self.offset):         
                    cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                    cv2.putText(frame,str(track_id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                    if track_id not in self.counter_down:
                        self.counter_down.append(track_id)  # get a list of the cars and buses which are entering the line red and exiting the line blue

            if track_id not in self.up: 
                if self.blue_line_y < (cy + self.offset) and self.blue_line_y > (cy - self.offset):
                    self.up[track_id]=cy  
            if track_id in self.up : 
                if self.red_line_y < (cy + self.offset) and self.red_line_y > (cy - self.offset): 
                    cv2.circle(frame,(cx,cy),4,(0,0,255),-1) 
                    cv2.putText(frame,str(track_id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                    if track_id not in self.counter_up:
                        self.counter_up.append(track_id) 
                        print(track_id)
                
    def plot_results(self,frame):
        text_color = (255,255,255)  # white color for text
        red_color = (0, 0, 255)  # (B, G, R)   
        blue_color = (255, 0, 0)  # (B, G, R)
        green_color = (0, 255, 0)  # (B, G, R)  


        cv2.line(frame,(170,700),(1850,700),red_color,3)  #  starting cordinates and end of line cordinates
        cv2.putText(frame,('red line'),(172,700),cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

        cv2.line(frame,(8,750),(1900,750),blue_color,3)  # seconde line
        cv2.putText(frame,('blue line'),(8,750),cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)    


        downwards = (len(self.counter_down))
        upwards = (len(self.counter_up))

        cv2.rectangle(frame, 
                      (int(1820 * 0.5 ) - 10, self.red_line_y + 40), 
                      (int(1820 * 0.5 ) + 200, self.red_line_y + 120), 
                      (0, 255, 255), 
                      cv2.FILLED)

        cv2.putText(img=frame, 
                    text=f'In:{downwards}', 
                    org=(int(1820 * 0.5 ), self.red_line_y + 90), 
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                    fontScale=1, 
                    color=(0, 0, 0), 
                    thickness=2)

        cv2.rectangle(frame, 
                      (int(1800 * 0.5 ) - 10, self.red_line_y - 100), 
                      (int(1800 * 0.5 ) + 200, self.red_line_y - 20), 
                      (0, 255, 255), 
                      cv2.FILLED)

        cv2.putText(img=frame, 
                    text=f'Out:{upwards}', 
                    org=(int(1800 * 0.5 ), self.red_line_y - 50), 
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                    fontScale=1, 
                    color=(0, 0, 0), 
                    thickness=2) 



        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
        return frame_bgr 

    def process_video(self,video,write_video=True,output_video_name=None):
        cap,width,height,fps=self.capture(video)
        if write_video:
            self.fetch_video_writer(output_video_name,width,height,fps)
        while True:
            ret,frame = cap.read()
            if not ret:
                break
            frame=self.resize_frame(frame,dim=(1920,1080))
            detections,conf,frame_updated=self.get_detections(frame)
            self.count(detections,conf,frame_updated)
            frame_bgr=self.plot_results(frame_updated)
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
            frame_bgr=cv2.resize(frame_bgr,(width,height))
            self.write_video(frame_bgr)
        cap.release()
        self.output_video.release() 
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

st.title("license plate detection using yolo")

uploaded_file= st.file_uploader("upload an image or video : ", type= ["jpg","jpeg","png","mp4","mkv"])

model= YOLO("/Users/bansalshrivastava/Desktop/object_detection/license_plate_detection_model.pt")



def process_media(input_path,output_path):
    file_extention= os.path.splitext(input_path)[1].lower()
    if file_extention in [".mkv",".mp4"]:
        return pred_and_plot_video(input_path,output_path)
    elif file_extention in [".jpg",".jpeg",".png"]:
        return pred_and_save_image(input_path,output_path)
    else:
        st.error(f"unsupported_file_type: {file_extention}")
        return None



def pred_and_save_image(path_test_car,output_image_path):
    results= model.predict(path_test_car, device= 'cpu')
    image= cv2.imread(path_test_car)
    image= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    for result in results:
        for box in result.boxes:
            x1,y1,x2,y2= map(int,box.xyxy[0])
            confidence=box.conf[0]
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(image,f'{confidence*100:.2f}%',
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
    image= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_image_path,image)
    return output_image_path



def pred_and_plot_video(video_path,output_path):
    cap= cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f'getting error in video opening:{video_path}')
        return None
    frame_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=int(cap.get(cv2.CAP_PROP_FPS))
    fourcc= cv2.VideoWriter_fourcc(*'mp4v')
    out= cv2.VideoWriter(output_path,fourcc,fps,(frame_width,frame_height))
    while cap.isOpened():
        ret, frame=cap.read()
        if not ret:
            break
        rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results= model.predict(rgb_frame,device='cpu')
        for result in results:
            for box in result.boxes:
                x1,y1,x2,y2= map(int,box.xyxy[0])
                confidence=box.conf[0]
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,f'{confidence*100:.2f}%',
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
        out.write(frame)
    cap.release()
    out.release()
    return output_path



if uploaded_file is not None:
    input_path = os.path.join("temp", uploaded_file.name)
    output_path = os.path.join("temp", f"output_{uploaded_file.name}")
    try:
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write("Processing...")
        result_path = process_media(input_path, output_path)
        if result_path:
            if input_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_file = open(result_path, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
            else:
                st.image(result_path)
    except Exception as e:
        st.error(f"Error uploading or processing file: {e}")

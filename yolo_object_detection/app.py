# import necessary libraries here
import cv2
import streamlit as st
import os
import json
from ultralytics import YOLO

with open('path_file.json','r') as out:
    loaded_data = json.load(out)

MODEL_PATH = loaded_data['model_path']
INPUT_PATH = loaded_data['input_path']
OUTPUT_PATH = loaded_data['output_path']
def app():
    '''
    Hosts Yolo Detection streamlit application

    Returns: None
    '''

    #Change model if required
    model = YOLO('yolov8n.pt')
    supported_objects = model.names
    st.title('YOLO Object Detection')
    st.subheader('Powered by Ultralytics')

    #Form to recieve Input
    with st.form('Input', clear_on_submit=True):
        uploaded_file = st.file_uploader('Upload Your Video:', type=['mp4'])
        class_value = st.multiselect('Classes supported', options=supported_objects.values(),
                                     default=list(supported_objects.values())[0])
        slider_value = st.slider('Confidence', min_value=0.0, max_value=1.0, step=0.1)
        submit = st.form_submit_button()

        if submit:
            name = uploaded_file.name
            contents = uploaded_file.read()
            with open(os.path.join(os.getcwd(), INPUT_PATH, name), 'wb') as out:
                out.write(contents)

            video_data = cv2.VideoCapture(os.path.join(os.getcwd(), INPUT_PATH, name))
            video_objects = []

            frame_height, frame_width = int(video_data.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
                video_data.get(cv2.CAP_PROP_FRAME_WIDTH))
            fps_value = int(video_data.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'h264')

            output_path = os.path.join(os.getcwd(), OUTPUT_PATH)
            file_path = os.path.join(output_path, name.split('.')[0] + f'{class_value[0]}_detected.mp4')

            if os.path.exists(output_path):
                video_writer = cv2.VideoWriter(file_path, codec, fps_value, (frame_width, frame_height))
            else:
                os.makedirs(output_path)
                video_writer = cv2.VideoWriter(file_path, codec, fps_value, (frame_width, frame_height))

            #Process the video for selected class
            with st.spinner(f'Finding {class_value[0]} in your Video'):
                while True:
                    ret, frame = video_data.read()

                    if not ret:
                        break

                    results = model(frame)

                    for result in results[0].boxes.data:
                        score = round(float(result[4]), 2)
                        if score >= slider_value:
                            # video_objects[result[0].boxes.cls[0]] = video_objects[result[0].boxes.data[0]]
                            x0, y0 = (int(result[0]), int(result[1]))
                            x1, y1 = (int(result[2]), int(result[3]))
                            cls_value = int(result[5])

                            if model.names[cls_value] not in video_objects:
                                video_objects.append(model.names[cls_value])

                            if model.names[cls_value] in class_value:
                                cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                                cv2.putText(frame, model.names[cls_value], (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (255, 0, 0), 2)
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                video_writer.write(frame)

                video_data.release()
                video_writer.release()
                st.info('Video Has been processed')

            st.video(file_path)


if __name__ == "__main__":
    app()
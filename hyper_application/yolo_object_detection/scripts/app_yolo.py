import streamlit as st
from streamlit.components.v1 import iframe
import cv2
import os
import json
from ultralytics import YOLO

# Streamlit App
def main():

    #set tab name
    st.set_page_config(page_title="Portfolio Demo")

    # Set up sidebar navigation
    st.sidebar.title("Navigation")
    selected_project = st.sidebar.radio("Select a Project", ["Portfolio", "YOLO Object Detection",
                                                             "Anomaly Detection with XGBoost"])

    projects = {
        "Adversarial Robustness - GenAI": "http://35.188.103.170:83",  # Replace with your GKE URLs
        "Automating Legal Contracts (NLP BERT Transformers)": "http://34.134.205.130:84/",
        "SimCLR Contrastive Learning": "http://34.68.153.135:82/",
        # "Anomaly Detection using XGBoost":"http://35.222.130.163:80"
    }

    # Portfolio Section
    if selected_project == "Portfolio":
        # Header and description
        st.title("Portfolio Projects")

        st.markdown(
            """
            **GitHub Repository / Portfolio website**  
            Check out my entire project collection on [GitHub](https://github.com/arunprasathjayaprakash/portfolio.git), [Portfolio](https://arun826jp.wixsite.com/my-site).
            """,
            unsafe_allow_html=True
        )

        st.info(
            """
            **All Projects are hosted and scaled with Google Kubernetes Engine, containerized using Docker,**
            producing a robust showcase of skills below.

            **Skill Set**:
            1. Machine Learning
            2. Deep Learning (Natural Language Processing, Transformers (BERT), Neural Networks)
            3. Extract Transform Load Pipelines (ETL)
            4. Kubernetes
            5. Docker
            6. Cloud-based Deployment (GCP Kubernetes, Container Registry)
            7. GenAI (Adversarial Attacks using PyTorch)

            The projects are structured to showcase not only machine learning capabilities but also MLops functionalities, 
            providing scalable solutions to complex challenges.

            ** For all Examples just run submit to see them in action data is preloaded for demo purposes**
            """
        )

        

        # Embed project-specific apps
        selected_project_name = st.sidebar.radio("Select a Specific Project", list(projects.keys()))
        st.markdown(f"### {selected_project_name}")
        iframe_src = projects[selected_project_name]
        if iframe_src:
            iframe(src=iframe_src, width=1000, height=4000)
        else:
            st.warning("This project currently does not have an external deployment link.")

    # YOLO Object Detection Section
    elif selected_project == "YOLO Object Detection":

        _ = st.sidebar.radio("Select a Specific Project", list(projects.keys()))

        # Load paths from a JSON file
        with open(os.path.join(os.getcwd(),'feature_store/path_file.json'), 'r') as file:
            loaded_data = json.load(file)

        MODEL_PATH = loaded_data['model_path']
        INPUT_PATH = loaded_data['input_path']
        OUTPUT_PATH = loaded_data['output_path']

        # YOLO Model
        model = YOLO(os.path.join(os.getcwd(),MODEL_PATH))
        supported_objects = model.names

        st.title("YOLO Object Detection")
        st.subheader("Powered by Ultralytics")

        st.info("""
                    ### About This Project
                    This application uses the YOLOv8 model to perform object detection on videos. It allows you to detect and highlight specific objects in video files with adjustable confidence levels for precise results.

                    **Key Features**:
                    - **Video Object Detection**: Upload video files to detect and highlight supported objects.
                    - **YOLOv8 Model**: Powered by the Ultralytics YOLOv8 model, renowned for its high speed and accuracy.
                    - **Customizable Options**:
                    - Select specific object classes for detection.
                    - Adjust confidence levels to filter predictions.

                    **How It Works**:
                    1. Upload a video file in **MP4 format**.
                    2. Select the object classes you want to detect from the list of supported options.
                    3. Set a confidence threshold to fine-tune the detection accuracy.
                    4. The application processes the video and highlights the selected objects in real-time.

                    **Who Is It For?**
                    This tool is ideal for professionals in fields such as video analysis, surveillance, and content creation who need efficient and accurate object detection.
                       

                         ** Run Submit to see the models predictions for the sample video **    
                    """)

        try:
#
            #Change model if required
            model = YOLO('models/yolov8n.pt')
            supported_objects = model.names

            default_video = os.path.join(os.getcwd(),"input_data/video.mp4") # Replace with your video path

            st.title("Video Uploader")

            st.info("""
                        ** Default Video has already been loaded just hit submit and wait for the video to get processed **
                    """)

            # st.video(default_video)

            #Form to recieve Input
            with st.form('Input', clear_on_submit=True):
                uploaded_file = st.file_uploader('Upload Your Video:', type=['mp4'],
                                                )
                
                class_value = st.multiselect('Classes supported', options=supported_objects.values(),
                                            default=list(supported_objects.values())[0])
                slider_value = st.slider('Confidence', min_value=0.0, max_value=1.0, step=0.1)
                submit = st.form_submit_button()

                if submit:
                    if uploaded_file:
                        name = uploaded_file.name
                        contents = uploaded_file.read()
                        with open(os.path.join(os.getcwd(), INPUT_PATH, name), 'wb') as out:
                            out.write(contents)
                    else:
                        name = 'video.mp4'


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
                
        except Exception as e:
            raise e
    elif selected_project == "Anomaly Detection with XGBoost":
        st.info("**Container is down for bug fixes will be up soon**")
if __name__ == "__main__":
    main()

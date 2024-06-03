# YOLO Object Detection App

This repository contains a Streamlit application for object detection in videos using YOLO (You Only Look Once) models. The app leverages the Ultralytics YOLO implementation for efficient and accurate object detection.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/arunprasathjayaprakash/portfolio/tree/d7ca243804b3e02039a654fbaa92784a10816adf/yolo_object_detection
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Set up the `path_file.json` file with the paths for your model, input, and output directories. An example `path_file.json`:

    ```json
    {
        "model_path": "path/to/your/yolo/model",
        "input_path": "Input Data",
        "output_path": "Output"
    }
    ```

2. Start the Streamlit application:

    ```bash
    streamlit run Scripts/app.py
    ```

3. Use the Streamlit interface to:
    - Upload a video file for object detection.
    - Select the classes to detect from the available options.
    - Set the confidence threshold for detection.
    - View the processed video with detected objects highlighted.

## Features

- **Video Upload**: Upload video files for processing.
- **Class Selection**: Choose specific object classes to detect in the video.
- **Confidence Slider**: Adjust the confidence threshold for object detection.
- **Processed Video**: View the video with detected objects highlighted.
- **Integration with Ultralytics YOLO**: Utilize the YOLO model from Ultralytics for object detection.

## Project Structure

```
Yolo Object detection/
│
├── Input Data/             # Directory to store input video files
├── Output/                 # Directory to store processed video files
├── Scripts/                # Directory containing all scripts and configurations
│   ├── app.py              # Main Streamlit application file
│   └── path_file.json      # JSON file containing paths for model, input, and output
├── requirements.txt        # Python dependencies
└── README.md               # Project README file
```

## Detailed Code Description

### `Scripts/app.py`

The main application file that uses Streamlit to create a web interface for object detection.

- **app()**: Main function to render the Streamlit application.
  - **model**: Initializes the YOLO model using Ultralytics.
  - **supported_objects**: Retrieves the list of supported object classes.
  - **Form to receive Input**: Streamlit form to upload video, select classes, and set confidence threshold.
  - **Video Processing**: Reads the uploaded video, processes it to detect objects, and saves the processed video.

### `Scripts/path_file.json`

A JSON configuration file that contains paths for the YOLO model, input directory, and output directory.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.

---
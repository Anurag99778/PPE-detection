# Personal Protective Equipment (PPE) Detector

This project is designed to detect whether a person is wearing a PPE kit or not using a trained YOLOv8 model. The model has been trained on a dataset of images and can be used to identify various components of a PPE kit in real-time.

## Table of Contents
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Usage](#usage)
- [Files](#files)
- [Project Structure](#project-structure)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/ppe-detector.git
    cd ppe-detector
    ```

2. Install the Ultralytics package:
    ```sh
    pip install ultralytics
    ```

## Training the Model

To train the YOLOv8 model, follow these steps:

1. Import YOLO from Ultralytics:
    ```python
    from ultralytics import YOLO
    ```

2. Run the following command to train the model. Make sure to replace the `data.yaml` path with the path to your dataset configuration file.
    ```sh
    yolo task=detect mode=train model=yolov8l.pt data=../content/drive/MyDrive/PPEDetection.v1-base_ver.yolov8/data.yaml epochs=50 imgsz=640
    ```

   - `task=detect`: Specifies the task as object detection.
   - `mode=train`: Sets the mode to training.
   - `model=yolov8l.pt`: Uses the pre-trained YOLOv8 large model.
   - `data=../content/drive/MyDrive/PPEDetection.v1-base_ver.yolov8/data.yaml`: Path to the data configuration file.
   - `epochs=50`: Number of training epochs.
   - `imgsz=640`: Image size for training.

By running these commands, you will obtain a `.pt` file, which is the trained model that you can use for running the PPE detection program.

## Usage

Once you have the trained model (`.pt` file), you can use it to detect PPE kits in images.

Here is an example of how to use the trained model to make predictions:

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('path/to/your/trained-model.pt')

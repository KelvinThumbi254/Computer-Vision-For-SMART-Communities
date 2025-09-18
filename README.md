# Smart Community Car Access System

A Computer Vision-based smart community access system built with Python and a Tkinter GUI. This project detects and reads car number plates, checks if the vehicle is registered as part of the community, and determines whether to allow entry.

## Features

License Plate Detection & Recognition – Uses computer vision (OpenCV) to detect and read car number plates in real-time or from images.

Community Membership Check – Matches recognized plate numbers against a database of registered vehicles.

### Automated Entry Decision –

✅ If the car belongs to the community → allow entry automatically.

❌ If not → direct the driver to security personnel for manual verification.

Parking Fee Calculation – GUI allows security personnel to log visitor details and calculate parking fees.

User-Friendly GUI – Built with Tkinter for easy interaction and visualization.

## Tech Stack

Language: Python

GUI Library: Tkinter

Computer Vision: OpenCV (cv2)

Dependencies:

opencv-python

numpy

## How to Run

Clone this repository:

git clone https://github.com/KelvinThumbi254/Computer-Vision-For-Smart-Communities.git
cd Computer-Vision-For-Smart-Communities


## Install dependencies:

pip install opencv-python numpy


## Run the application:

python main.py


# Workflow

Capture / Upload Image – User uploads an image or uses live camera feed.

Plate Detection – The system detects the number plate using OpenCV.

Text Extraction – Recognizes plate number (OCR or template matching).

Verification – Checks against registered vehicle list.

Action –

Allow automatic entry if matched.

If not matched, security logs details and calculates parking fees.

## Future Improvements

Add database integration for persistent vehicle records.

Integrate live video feed for continuous real-time detection.

Use a more advanced OCR model for improved accuracy.

Generate visitor entry reports automatically.

# License

This project is open-source and free to use.

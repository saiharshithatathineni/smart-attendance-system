import face_recognition
import cv2
import os
from datetime import datetime
from flask import Flask, render_template, Response

app = Flask(__name__, template_folder="templates")

# Step 2: Create Dataset
# Update the dataset_path with the path to the folder containing the dataset images.
dataset_path = "C:/Users/satya/PycharmProjects/MLproject/dataset"

# Step 4: Face Encoding
# Load and encode the dataset images and store them in a dictionary.
def load_dataset():
    dataset1 = {}
    for folder_name in os.listdir(dataset_path):
        person_name = folder_name
        person_folder = os.path.join(dataset_path, folder_name)
        person_images = [os.path.join(person_folder, image) for image in os.listdir(person_folder)]
        person_encodings = [face_recognition.face_encodings(face_recognition.load_image_file(image))[0] for image in
                            person_images]
        dataset1[person_name] = person_encodings
    return dataset1

# Load the dataset at the start of the application.
dataset = load_dataset()

# Step 3: Face Detection using dlib
def detect_faces(frame):
    try:
        face_locations = face_recognition.face_locations(frame)
        return face_locations
    except Exception as e:
        # Log the error for debugging purposes
        with open('error_log.txt', mode='a') as error_log_file:
            error_log_file.write(str(datetime.now()) + " - " + str(e) + "\n")
        return []  # Return an empty list if face detection fails

# Step 5: Face Recognition
def recognize_faces(frame):
    face_encodings = face_recognition.face_encodings(frame)
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces([item for sublist in dataset.values() for item in sublist], face_encoding)
        name = "Unknown"
        if True in matches:
            matched_index = matches.index(True)
            name = list(dataset.keys())[matched_index]
        face_names.append(name)
    return face_names

# Step 6: Mark Attendance
log_file_path = 'C:/Users/satya/PycharmProjects/MLproject/log.csv'

def mark_attendance(name):
    ts = datetime.now()
    new_date = ts.strftime("%m-%d-%y")
    year = ts.strftime("%y")
    month = ts.strftime("%m")
    day = ts.strftime("%d")
    time1 = ts.strftime("%H:%M:%S")

    # Log the attendance to a CSV file (log.csv)
    with open(log_file_path, mode='a') as logFile:
        pos = logFile.tell()
        if pos == 0:
            logFile.write("Year,Month,Day,Time,Name,Attendance\n")
        info = "{},{},{},{},{},{}\n".format(year, month, day, time1, name, "Present")
        logFile.write(info)

# Step 7: Build the Flask App
@app.route("/", methods=['GET'])
def index():
    return render_template('home1.html')

@app.route('/home1', methods=['GET'])
def about():
    return render_template('home1.html')

@app.route("/take_attendance", methods=['GET'])
def take_attendance():
    return render_template('attendance_form.html')

# Step 8: Video Capture and Processing
def generate_frames():
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        face_locations = detect_faces(frame)  # Face detection
        face_names = recognize_faces(frame)  # Face recognition

        # Display the result on the video frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (6, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.6, (255, 255, 255), 1)

            # Step 9: Display Result and Log Attendance
            if name != "Unknown":
                mark_attendance(name)

        ret, jpeg_frame = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg_frame.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed', methods=['GET'])
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_stream', methods=['GET'])
def video_stream():
    return render_template('video_stream.html')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8000, debug=False)
    except Exception as e:
        print("Error occurred:", e)
        # Log the error for debugging purposes
        with open('error_log.txt', mode='a') as error_log_file:
            error_log_file.write(str(datetime.now()) + " - " + str(e) + "\n")

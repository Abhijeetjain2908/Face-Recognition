import cv2
import face_recognition
import os
import pandas as pd
from datetime import datetime, timedelta
import threading
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk

# Global variables
last_attendance_time = {}
attendance_records = {}
known_faces = []
known_names = []
face_id_mapping = {}

# Function to load known faces from the folder
def load_known_faces_from_folder(folder_path):
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)

        # Check if any face is detected in the image
        if len(face_encodings) > 0:
            face_encoding = face_encodings[0]
            known_faces.append(face_encoding)

            # Extract the person's name and face ID from the filename (assuming filename format: "John_Doe_123.jpg")
            name, ext = os.path.splitext(filename)
            if "_" in name:
                name, face_id = name.split("_")
                known_names.append(name)
                face_id_mapping[int(face_id)] = name
            else:
                print(f"Invalid filename format: {filename}")
        else:
            print(f"No face detected in image: {filename}")
            
# Function to mark attendance
def mark_attendance(name, face_id, status):
    # Get the current date and time
    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Check if the same person's attendance has been marked within the last 300 seconds (5 minutes)
    if name in last_attendance_time and datetime.now() - last_attendance_time[name] < timedelta(seconds=300):
        return

    # Update the last attendance time for the person
    last_attendance_time[name] = datetime.now()

    # Append the attendance record to the CSV file
    data = {'Name': [name], 'Face ID': [face_id], 'Status': [status], 'Date-Time': [date]}
    df = pd.DataFrame(data)

    if not os.path.isfile('attendance.csv'):
        df.to_csv('attendance.csv', index=False)
    else:
        df.to_csv('attendance.csv', mode='a', header=False, index=False)

    # Display a notification that attendance is marked
    notification_label.config(text=f"Attendance marked for {name} ({face_id}) - Status: {status} - Time: {date}")

# Function to register a new person
def register_new_person():
    global known_faces, known_names, face_id_mapping

    # Ask the user to enter the person's name
    name = simpledialog.askstring("Register New Person", "Enter the person's name:")

    if name is not None:
        # Check if the person's name already exists in the known_names list
        if name in known_names:
            messagebox.showwarning("Name Exists", "This name already exists. Please choose another name.")
            return

        # Ask the user to enter the Face ID for the person
        face_id = simpledialog.askinteger("Register New Person", "Enter the Face ID for the person:")

        if face_id is None or face_id <= 0:
            messagebox.showwarning("Invalid Face ID", "Please enter a valid positive integer as Face ID.")
            return

        # Capture the person's face using the camera feed
        video_capture = cv2.VideoCapture(0)

        while True:
            ret, frame = video_capture.read()

            # Locate faces in the image
            face_locations = face_recognition.face_locations(frame)

            if len(face_locations) == 0:
                # No face detected, continue to display the camera feed
                cv2.imshow("Capture Face", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Assuming only one face is captured
            top, right, bottom, left = face_locations[0]

            # Extract the face image
            face_image = frame[top:bottom, left:right]

            # Save the face image for future recognition
            face_image_path = f"registered_faces/{name}_{face_id}.jpg"
            cv2.imwrite(face_image_path, face_image)

            # Convert the face image to a PIL format and display it
            face_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
            face_pil = face_pil.resize((200, 200))
            face_tk = ImageTk.PhotoImage(face_pil)

            # Display the captured face with the name and Face ID
            register_label.config(image=face_tk, text=f"Name: {name}\nFace ID: {face_id}")
            register_label.image = face_tk

            # Add the registered face to the known_faces and known_names lists
            known_faces.append(face_recognition.face_encodings(face_image)[0])
            known_names.append(name)

            # Map the Face ID to the person's name for attendance marking
            face_id_mapping[face_id] = name

            # Stop capturing and exit the loop
            break

        # Release the video capture and close the window
        video_capture.release()
        cv2.destroyAllWindows()

# Function to recognize faces from the camera
def recognize_faces_from_camera():
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()

        # Initialize variables with default values
        name = "Unknown"
        face_id = None
        status = None

        # Locate faces in the image
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            # Compare the face with known faces
            matches = face_recognition.compare_faces(known_faces, face_encoding)

            if True in matches:
                # Find the index of the first match
                first_match_index = matches.index(True)
                name = known_names[first_match_index]

                # Get the face ID from the face_id_mapping
                face_id = [k for k, v in face_id_mapping.items() if v == name][0]

                # Determine the status (IN or OUT) based on previous attendance records (you can adjust this logic)
                status = "IN" if name not in attendance_records or attendance_records[name][-1] == "OUT" else "OUT"

                mark_attendance(name, face_id, status)

                # Draw a green rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Display "Verified" message and face_id below the face
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, "Verified", (left + 6, bottom - 30), font, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Face ID: {face_id}", (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
            else:
                # Reset name, face_id, and status when the person is unknown
                name = "Unknown"
                face_id = None
                status = None

                # Draw a red rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Display "Unknown" message below the face
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, "Unknown", (left + 6, bottom - 30), font, 0.5, (0, 0, 255), 1)

        # Convert the frame to PIL format
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Resize the frame to fit the GUI window
        frame_pil = frame_pil.resize((640, 480))

        # Convert PIL image to Tkinter PhotoImage
        frame_tk = ImageTk.PhotoImage(frame_pil)

        # Update the camera feed in the GUI
        canvas.create_image(0, 0, anchor=tk.NW, image=frame_tk)

        # Update the GUI with the current frame and notification
        notification_label.config(text=f"Attendance marked for {name} ({face_id}) - Status: {status}")

        # Update the canvas widget in the main GUI thread
        root.update()

    # Release the video capture and close the window
    video_capture.release()

# Function to start the face recognition and attendance process in a separate thread
def start_recognition_thread():
    # Load known faces and names from the folder
    folder_path = r"C:\Users\HP\Desktop\yolo1\New folder"
    load_known_faces_from_folder(folder_path)

    # Start the face recognition process in a separate thread
    threading.Thread(target=recognize_faces_from_camera).start()

# Create the GUI window
root = tk.Tk()
root.title("Face Recognition and Attendance")

# Create a Canvas widget to display the camera feed
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

# Create a label for notifications
notification_label = tk.Label(root, text="Waiting for recognition...", font=("Arial", 16))
notification_label.pack()

# Create a label for registering a new person
register_label = tk.Label(root, text="Register New Person", font=("Arial", 16))
register_label.pack()

# Create a button to register a new person
register_button = tk.Button(root, text="Register", command=register_new_person)
register_button.pack()

# Start the face recognition process in a separate thread
start_recognition_thread()

# Start the GUI event loop
root.mainloop()

import cv2
from pylsl import StreamInfo, StreamOutlet
import numpy as np
import time
import tkinter as tk
from tkinter import simpledialog
import keyboard
import os
from glob import glob
import re
import sys
import subprocess
import signal


print('Initializing LSL Video Stream... \n')
'''
experiment_phase = input("If recording for RESTING STATE, PRESS 1 then hit enter \nIf recording for STORY LISTENING task, PRESS 2 then hit enter \nIf recording for SOCIAL SCRIPT task, PRESS 3 then hit enter\n")

print("_____________________")
if re.fullmatch(r"[123]", experiment_phase):
    if int(experiment_phase) == 1:
        phase ="Rest"
        dir = 'C:/Users/Admin/Documents/CUNY_faceVideoFiles/CUNY_FACE_REST_VIDEO_FILES/'
        print("Begining the video stream for the Rest task")
    elif int(experiment_phase) == 2:
        phase ="StoryListening"
        dir = 'C:/Users/Admin/Documents/CUNY_faceVideoFiles/CUNY_FACE_STORYLISTEN_VIDEO_FILES/'
        print("Begining the video stream for the Story Listening task")
    elif int(experiment_phase) == 3:
        print("Begining the video stream for the Social Script task")
        phase = "SocialScript"
        dir = "C:/Users/Admin/Documents/CUNY_faceVideoFiles/CUNY_FACE_SOCIALSCRIPT_VIDEO_FILES/"
else:
    print('Input must be 1,2 or 3 . exiting...')
    sys.exit()
'''

dir = "C:/Users/Admin/Documents/CUNY_faceVideoFiles/FULL_PROTOCOL/"

print("_____________________")
print("Welcome to the CUNY FACE EXPERIMENT")

print("_____________________")

subject_id = input('Please enter subject ID:')

if re.fullmatch(r"\d+", subject_id):
    print(".............")

    print(f"Next, adjust the camera angle to capture subject {subject_id}")
else:
    print('Subject ID must be an integer. exiting...')
    sys.exit()




def get_subject_id():
    root = tk.Tk()
    root.withdraw()

    user_input = simpledialog.askstring("Input", "Please Enter the Subject ID:")
    '''
    radio_window = tk.Toplevel(root)
    radio_window.title("Experiment Information")

    selected_option = tk.StringVar()
    options = ['Resting State', 'Story Listening', 'Social Script']
    for option in options:
        tk.Radiobutton(radio_window,text=option, variable=selected_option. valu)
    '''
    if user_input is not None:
        print(f"Now Recording subject:{user_input}. Starting stream, please wait")
    else:
        print("User Canceled the input dialog box")
    return user_input


def stream_video(subject_id, directory, process):

    # Set up video capture from webcam  
    cap = cv2.VideoCapture(0)

    # Define stream information
    stream_name = "WebcamStream"
    stream_id = "WebcamStreamID"
    stream_channels = 4  #921601  # RGB channels
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
   
    print(cap.get(cv2.CAP_PROP_FPS))
    # Create LSL stream outlet
    info = StreamInfo(stream_name, 'video', stream_channels, fps, 'string', stream_id)
    outlet = StreamOutlet(info)
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    out = cv2.VideoWriter(directory+subject_id+f"_CUNY_Face_Full_Protocol.avi", fourcc, 30, (int(width), int(height)))

    frameCounter = 1
    print("Begin the PsychoPy Experiment, then Press the space bar to begin recording")
    keyboard.wait('space')
    # Start streaming
    print("RECORDING>>>>>")
    while True:
        timestamp_system_pre = time.time()
        # Capture frame-by-frame
        ret, frame = cap.read()
        timestamp_video = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp_system_post = time.time()
        # If frame is read correctly, send it via LSL stream
        if ret:
            # Convert frame to bytes and send it
            outlet.push_sample([
                str(frameCounter), 
                str(timestamp_system_pre), 
                str(timestamp_video), 
                str(timestamp_system_post)
                ])
            out.write(frame)
            #outlet.push_sample(np.insert(frame.flatten().tolist(), 0, time.time()))
            
            # Display the resulting frame
            cv2.imshow('Webcam Stream: Press q to exit', frame)
            frameCounter += 1
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('VIDEO RECORDING STOPPED')
            print("EYE TRACKING STREAM STILL RUNNING: press CTRL-C repeatedly to stop it...")
            #process.terminate()
            process.wait()

            break
    
    # Release the capture and destroy the window 
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def check_FOV():

    # Set up video capture from webcam  
    cap = cv2.VideoCapture(0)

    # Define stream information
    stream_name = "WebcamStream"
    stream_id = "WebcamStreamID"
    stream_channels = 4  #921601  # RGB channels
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
   
    
    # Start streaming
    print("Check WebCam Field of View, press \"q\" when ready....")
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, send it via LSL stream
        if ret:
            # Display the resulting frame
            cv2.imshow('Webcam Stream: Press q to exit', frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Field of View has been adjusted...')
            break
 
    # Release the capture and destroy the window 
    cap.release()
    cv2.destroyAllWindows()

def check_file(dir, fname):
    """ Checks the specified directory for a file with the given name and renames it with the suffix '_old' if found. 
    
    Parameters: 
    directory (str): The path to the directory to search in.   
    filename (str): The name of the file to check for and rename. 
    """ 
    subid = fname.split('_')[0]
    fpath = os.path.join(dir, fname)
    if os.path.isfile(fpath):
        sub_files = glob(os.path.join(dir, f"*{subid}*"))
        if len(sub_files) == 1:
            new_file_name = fname.split('.')[0] + "_old.avi"
            new_path = os.path.join(dir, new_file_name)
            # get original timestamp metadata
            orig_modtime = os.path.getmtime(sub_files[0])
            os.rename(sub_files[0], new_path)
            # after renaming set back with original timestamp metadata
            os.utime(new_path, (orig_modtime, orig_modtime)) 
            return
        elif len(sub_files) > 1:
            for i, old_file_path in enumerate(reversed(sub_files)):
                new_file_path = os.path.join(dir, fname.split('.')[0]+f"_old{len(sub_files) - i}.avi")
                orig_modtime = os.path.getmtime(old_file_path)
                os.rename(old_file_path, new_file_path)
                os.utime(new_file_path, (orig_modtime, orig_modtime))
            return
        else:
            return
    else:
        return

def start_tobii():


    ET_script = "C:/Users/Admin/Desktop/tobiilsl.py"

    python_executable = "C:/Program Files/PsychoPy/python.exe"
    command = [python_executable, ET_script, subject_id]
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    #process = subprocess.Popen(command) # with this we would see the packets transfers from th stdout
    print(process.stdout)
    if process.stderr:
        print(process.stderr)
    return process
    
def signal_handler(sig, frame):
    print('stopping the tobii script....')
    if process:
        process.terminate()
    sys.exit(0)


#subject_id = get_subject_id()
check_file(dir, subject_id+f"_CUNY_Face_Full_Protocol.avi" )

check_FOV()

#Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

process = start_tobii()


# Start video streaming
stream_video(subject_id=subject_id, directory=dir, process=process)
#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Wed Oct 20 11:50:46 2021

 

@author: jcloud

"""

 

import pyaudio
import wave
from pylsl import StreamInfo, StreamOutlet
import time, socket
import numpy as np
import argparse
import sys, os
import re
from glob import glob

#set arguments - FILENAME, needs .wav extension

print('Initializing LSL Microphone Stream... \n')
experiment_phase = input("If recording for STORY LISTENING task, PRESS 2 then hit enter\nIf recording for SOCIAL SCRIPT task, PRESS 3 then hit enter\n")

print("_____________________")
if re.fullmatch(r"[23]", experiment_phase):
    if int(experiment_phase) == 2:
        phase ="StoryListening"
        dir = 'CUNY_FACE_STORYLISTEN_AUDIO_FILES/'
        print("Begining the microphone stream for the Story Listening task")
    elif int(experiment_phase) == 3:
        print("Begining the microphone stream for the Social Script task")
        phase = "SocialScript"
        dir = "CUNY_FACE_SOCIALSCRIPT_AUDIO_FILES/"
else:
    print('Input must be 1 or 2. exiting...')
    sys.exit()

print("_____________________")
subject_id = input('Please enter subject ID:')

if re.fullmatch(r"\d+", subject_id):
    print(f"Starting recording of subject {subject_id}")
else:
    print('Subject ID must be an integer. exiting...')
    sys.exit()

#parser = argparse.ArgumentParser(description='')
#parser.add_argument('--filename', dest='filename', type=str, help='name of output wav file', required = True)
#args = parser.parse_args()

#set up
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
WAVE_OUTPUT_NAME = f"{subject_id}_CUNY_FaceProject_{phase}.wav" #args.filename
CHUNK = int(RATE * .1)
 
p = pyaudio.PyAudio()
x = p.get_device_count()

for i in range(x):

    dev = p.get_device_info_by_index(i)
    print("DEV: ",dev)

    if 'RÃ˜DE NT-USB' in dev['name'] and dev['maxInputChannels'] != 0:

        break

       

if 'RÃ˜DE NT-USB' not in dev['name']:

    print('\n\n WARNING \n\n WARNING \n\n WARNING \n\n SOUND NOT CONFIGURED CORRECTLY \n\n GRACE EXIT AND CHANGE THE SOUND SETTINGS \n\n')

 

print('Recording from ' + dev['name'])

 

stream = p.open(format = FORMAT,

                channels = CHANNELS,

                rate = RATE,

                input = True,

                frames_per_buffer = CHUNK,

                input_device_index = i)

 

channels = CHANNELS

fps = RATE

 

info = StreamInfo('Audio', 'AudioCapture', channels, fps, 'int16', 'audio-' + socket.gethostname())

outlet = StreamOutlet(info, CHUNK, 360)

 

frames = []

 
print("RECORDING>>>>>")
print("_____________________")
print("press Ctrl + c to end...")

while True:

    try:

        data = stream.read(CHUNK) # byte string containing the raw audio data

        # nump = np.fromstring(data, dtype=np.int16) # changed because of depreciation warning, BG
        
        nump = np.frombuffer(data, dtype=np.int16)
        outlet.push_chunk(nump)
        frames.append(data)

    except KeyboardInterrupt:

        break

    except:

        print('Error')

   

print('Audio Closed')

stream.stop_stream()

stream.close()

p.terminate()
####
#CHECKING FOR DUPLICATE FILE
def check_file(dir, fname):
    subid = fname.split('_')[0]
    fpath = os.path.join(dir, fname)
    if os.path.isfile(fpath):
        sub_files = glob(os.path.join(dir, f"*{subid}*"))
        if len(sub_files) == 1:
            new_file_name = fname.split('.')[0] +'_old.wav'
            new_path = os.path.join(dir, new_file_name)
            # get metadata timestamp of original
            orig_modtime = os.path.getmtime(sub_files[0])
            os.rename(sub_files[0], new_path)
            # reset the metadata timestamp to original
            os.utime(new_path, (orig_modtime, orig_modtime))
            return
        elif len(sub_files) > 1:
            for i, old_file_path in enumerate(reversed(sub_files)):
                new_file_path = os.path.join(dir, fname.split('.')[0]+f"_old{len(sub_files) - i}.wav")
                orig_modtime = os.path.getmtime(old_file_path)
                os.rename(old_file_path, new_file_path)
                os.utime(new_file_path, (orig_modtime, orig_modtime))
            return
        else:
            return
    else:
        return
    

check_file(dir, WAVE_OUTPUT_NAME)
 

wf = wave.open(os.path.join(dir, WAVE_OUTPUT_NAME), 'wb')

wf.setnchannels(CHANNELS)

wf.setsampwidth(p.get_sample_size(FORMAT))

wf.setframerate(RATE)

wf.writeframes(b''.join(frames))

wf.close()
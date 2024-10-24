#
#
#
################################
# SETUP HERE
#

license_file = ""#"license_file"

# DONT CHANGE BELOW



################################
# Preface here
#
# from psychopy import prefs, visual, core, event, monitors, tools, logging
import numpy as np
import tobii_research as tr
import time
import random
import os
import pylsl as lsl
import sys
import pandas as pd
import argparse
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('value', type=str)

args = parser.parse_args()

# Find Eye Tracker and Apply License (edit to suit actual tracker serial no)
ft = tr.find_all_eyetrackers()
if len(ft) == 0:
    print("No Eye Trackers found!?")
    exit(1)

# Pick first tracker
mt = ft[0]
print("Found Tobii Tracker at '%s'" % (mt.address))
print(f"Subject from Tobii script is {args.value}")

# Apply license
if license_file != "":
    with open(license_file, "rb") as f:
        license = f.read()

        res = mt.apply_licenses(license)
        if len(res) == 0:
            print("Successfully applied license from single key")
        else:
            print("Failed to apply license from single key. Validation result: %s." % (res[0].validation_result))
            exit
else:
    print("No license file installed")

channels = 31 # count of the below channels, incl. those that are 3 or 2 long
gaze_stuff = [
    ('device_time_stamp', 1),

    ('left_gaze_origin_validity',  1),
    ('right_gaze_origin_validity',  1),

    ('left_gaze_origin_in_user_coordinate_system',  3),
    ('right_gaze_origin_in_user_coordinate_system',  3),

    ('left_gaze_origin_in_trackbox_coordinate_system',  3),
    ('right_gaze_origin_in_trackbox_coordinate_system',  3),

    ('left_gaze_point_validity',  1),
    ('right_gaze_point_validity',  1),

    ('left_gaze_point_in_user_coordinate_system',  3),
    ('right_gaze_point_in_user_coordinate_system',  3),

    ('left_gaze_point_on_display_area',  2),
    ('right_gaze_point_on_display_area',  2),

    ('left_pupil_validity',  1),
    ('right_pupil_validity',  1),

    ('left_pupil_diameter',  1),
    ('right_pupil_diameter',  1)
]
    

def unpack_gaze_data(gaze_data):
    x = []
    for s in gaze_stuff:
        d = gaze_data[s[0]]
        if isinstance(d, tuple):
            x = x + list(d)
        else:
            x.append(d)
    return x

last_report = 0
N = 0

et_dat = []

def gaze_data_callback(gaze_data):
    '''send gaze data'''

    '''
    This is what we get from the tracker:

    device_time_stamp

    left_gaze_origin_in_trackbox_coordinate_system (3)
    left_gaze_origin_in_user_coordinate_system (3)
    left_gaze_origin_validity
    left_gaze_point_in_user_coordinate_system (3)
    left_gaze_point_on_display_area (2)
    left_gaze_point_validity
    left_pupil_diameter
    left_pupil_validity

    right_gaze_origin_in_trackbox_coordinate_system (3)
    right_gaze_origin_in_user_coordinate_system (3)
    right_gaze_origin_validity
    right_gaze_point_in_user_coordinate_system (3)
    right_gaze_point_on_display_area (2)
    right_gaze_point_validity
    right_pupil_diameter
    right_pupil_validity

    system_time_stamp
    '''


    # for k in sorted(gaze_data.keys()):
    #     print(' ' + k + ': ' +  str(gaze_data[k]))

    try:
        global last_report
        global outlet
        global N
        global halted

        sts = gaze_data['system_time_stamp'] / 1000000.

        outlet.push_sample(unpack_gaze_data(gaze_data), sts)
        et_dat.append(unpack_gaze_data(gaze_data))

        if sts > last_report + 5:
            sys.stdout.write("%14.3f: %10d packets\r" % (sts, N))
            last_report = sts
           # print("%14.3f: %10d packets\r" % (sts, N))
        N += 1

        # print(unpack_gaze_data(gaze_data))
    except:
        print("Error in callback: ")
        print(sys.exc_info())

        halted = True




def start_gaze_tracking():
    mt.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)
    return True


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
                new_file_path = os.path.join(dir, fname.split('.')[0]+f"_old{len(sub_files) - i}.csv")
                orig_modtime = os.path.getmtime(old_file_path)
                os.rename(old_file_path, new_file_path)
                os.utime(new_file_path, (orig_modtime, orig_modtime))
            return
        else:
            return
    else:
        return


def end_gaze_tracking(et_dat):
    mt.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
    df = pd.DataFrame(et_dat)

    check_file(dir="C:/Users/Admin/Desktop/CUNY_ET_RAW/", fname="sub-P"+ args.value +"_ses-S001_task-CUNY_run-001_eyes.csv")

    df.to_csv("C:/Users/Admin/Desktop/CUNY_ET_RAW/sub-P"+ args.value +"_ses-S001_task-CUNY_run-001_eyes.csv", sep=',')
    return True

halted = False




# Set up lsl stream
def setup_lsl():
    global channels
    global gaze_stuff

    info = lsl.StreamInfo('Tobii', 'ET', channels, 90, 'float32', mt.address)

    info.desc().append_child_value("manufacturer", "Tobii")
    channels = info.desc().append_child("channels")
    cnt = 0
    for s in gaze_stuff:
        if s[1]==1:
            cnt += 1
            channels.append_child("channel") \
                    .append_child_value("label", s[0]) \
                    .append_child_value("unit", "device") \
                    .append_child_value("type", 'ET')
        else:
            for i in range(s[1]):
                cnt += 1
                channels.append_child("channel") \
                        .append_child_value("label", "%s_%d" % (s[0], i)) \
                        .append_child_value("unit", "device") \
                        .append_child_value("type", 'ET')

    outlet = lsl.StreamOutlet(info)

    return outlet

outlet = setup_lsl()

# Main loop; run until escape is pressed
print("%14.3f: LSL Running; press CTRL-C repeatedly to stop" % lsl.local_clock())
start_gaze_tracking()
try:
    while not halted:
        time.sleep(1)
        keys = ()  # event.getKeys()
        if len(keys) != 0:
            if keys[0]=='escape':
                
                halted = True

        if halted:
            break

        # print(lsl.local_clock())

except:
    print("Halting...")


print("terminating tracking now")
end_gaze_tracking(et_dat)


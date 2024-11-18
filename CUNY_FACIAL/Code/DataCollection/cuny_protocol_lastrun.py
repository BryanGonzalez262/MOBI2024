#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.1.1),
    on November 18, 2024, at 15:16
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard
from psychopy.hardware import camera
from psychopy.sound import microphone

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.1.1'
expName = 'audio_task'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': '',
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = False
_loggingLevel = logging.getLevel('warning')
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
    # override logging level
    _loggingLevel = logging.getLevel(
        prefs.piloting['pilotLoggingLevel']
    )

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/sub-%s_task-CUNY_run-001_behavior' % (expInfo['participant'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Admin\\Documents\\PsychoPy_Experiments\\cuny_facial_protocol\\cuny_protocol_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(_loggingLevel)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=_loggingLevel)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1920, 1080], fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0.0039, 0.0039, 0.0039], colorSpace='rgb',
            backgroundImage='', backgroundFit='fill',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0.0039, 0.0039, 0.0039]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'fill'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = True
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('start_task') is None:
        # initialise start_task
        start_task = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='start_task',
        )
    if deviceManager.getDevice('rst_key_start') is None:
        # initialise rst_key_start
        rst_key_start = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='rst_key_start',
        )
    if deviceManager.getDevice('audio_next') is None:
        # initialise audio_next
        audio_next = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='audio_next',
        )
    if deviceManager.getDevice('audio_next2') is None:
        # initialise audio_next2
        audio_next2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='audio_next2',
        )
    if deviceManager.getDevice('audio_next3') is None:
        # initialise audio_next3
        audio_next3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='audio_next3',
        )
    # create speaker 'audio_file'
    deviceManager.addDevice(
        deviceName='audio_file',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=11.0
    )
    if deviceManager.getDevice('impedance_end') is None:
        # initialise impedance_end
        impedance_end = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='impedance_end',
        )
    # create speaker 'social_end_sound'
    deviceManager.addDevice(
        deviceName='social_end_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=11.0
    )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "setup" ---
    text = visual.TextStim(win=win, name='text',
        text='Please wait while we set up everything.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=[0.9137, -0.3647, 0.9765], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from start_lsl
    import pylsl
    from pylsl import StreamInfo, StreamOutlet
    
    info = StreamInfo(name="Stimuli_Markers", type="Markers", channel_count=1, channel_format="float32", source_id="PsychoPy Markers")
    outlet = StreamOutlet(info)
    start_task = keyboard.Keyboard(deviceName='start_task')
    
    # --- Initialize components for Routine "resting_state_welcome" ---
    rst_welcome = visual.TextStim(win=win, name='rst_welcome',
        text='Hi! In this task, please focus on the screen and try to remain relaxed and avoid sudden movements.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=[0.9137, -0.3647, 0.9765], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    rst_key_start = keyboard.Keyboard(deviceName='rst_key_start')
    
    # --- Initialize components for Routine "resting_state" ---
    rst_screen = visual.TextStim(win=win, name='rst_screen',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "audio_welcome" ---
    welcome_instr = visual.TextStim(win=win, name='welcome_instr',
        text='Hi! In this task, you will listen to a selection of short stories. Pay attention to each story.\n\nAfter each story is played, you will answer a few questions about it.\n\n',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=[0.9137, -0.3647, 0.9765], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    audio_next = keyboard.Keyboard(deviceName='audio_next')
    
    # --- Initialize components for Routine "relax" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='Just relax and be natural. Please avoid sudden movements.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=[0.9137, -0.3647, 0.9765], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    audio_next2 = keyboard.Keyboard(deviceName='audio_next2')
    
    # --- Initialize components for Routine "audio_instructions" ---
    disp_audio_name = visual.TextStim(win=win, name='disp_audio_name',
        text='Now, you will listen to the story:',
        font='Open Sans',
        pos=(0, 0.2), height=0.05, wrapWidth=None, ori=0.0, 
        color=[0.9137, -0.3647, 0.9765], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    audio_name = visual.TextStim(win=win, name='audio_name',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.04, wrapWidth=None, ori=0.0, 
        color=[0.9137, -0.3647, 0.9765], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    emoji = visual.ImageStim(
        win=win,
        name='emoji', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.2), size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    audio_next3 = keyboard.Keyboard(deviceName='audio_next3')
    
    # --- Initialize components for Routine "audio_rest" ---
    rest_before_audio = visual.TextStim(win=win, name='rest_before_audio',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=[-0.9137, 0.3725, -0.9216], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "audio" ---
    audio_crosshair = visual.TextStim(win=win, name='audio_crosshair',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=[-0.9137, 0.3725, -0.9216], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    audio_file = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='audio_file',    name='audio_file'
    )
    audio_file.setVolume(1.0)
    
    # --- Initialize components for Routine "after_audio_instructions" ---
    inst_after = visual.TextStim(win=win, name='inst_after',
        text='You have just listened to:',
        font='Open Sans',
        pos=(0, 0.3), height=0.05, wrapWidth=None, ori=0.0, 
        color=[0.9137, -0.3647, 0.9765], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    audio_name2 = visual.TextStim(win=win, name='audio_name2',
        text='',
        font='Open Sans',
        pos=(0, 0.2), height=0.04, wrapWidth=None, ori=0.0, 
        color=[0.9137, -0.3647, 0.9765], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    emoji2 = visual.ImageStim(
        win=win,
        name='emoji2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    inst_after2 = visual.TextStim(win=win, name='inst_after2',
        text='Now, you will answer questions about this story. Try your best to answer each question as accurately as possible.',
        font='Open Sans',
        pos=(0, -0.3), height=0.05, wrapWidth=None, ori=0.0, 
        color=[0.9137, -0.3647, 0.9765], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    next_3 = visual.ButtonStim(win, 
        text='Next', font='Arvo',
        pos=(0.5, -0.4),
        letterHeight=0.05,
        size=(0.3, 0.1), borderWidth=0.0,
        fillColor=[-0.4510, 0.0196, 0.4118], borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='next_3',
        depth=-4
    )
    next_3.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "questions" ---
    question = visual.TextStim(win=win, name='question',
        text='',
        font='Open Sans',
        pos=(0, 0.4), height=0.05, wrapWidth=None, ori=0.0, 
        color=[0.9137, -0.3647, 0.9765], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    slider = visual.Slider(win=win, name='slider',
        startValue=4, size=(1.5, 0.1), pos=(0, -0.1), units=win.units,
        labels=['Strongly\nDisagree', 'Disagree','Somewhat\nDisagree', 'Neutral', 'Somewhat\nAgree', 'Agree', 'Strongly\nAgree'], ticks=(1, 2, 3, 4, 5, 6, 7), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor=[0.9137, -0.3647, 0.9765], markerColor=[0.9137, -0.3647, 0.9765], lineColor=[0.9137, -0.3647, 0.9765], colorSpace='rgb',
        font='Open Sans', labelHeight=0.04,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    disagree = visual.ImageStim(
        win=win,
        name='disagree', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.75, 0.1), size=(0.22, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    agree = visual.ImageStim(
        win=win,
        name='agree', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0.75, 0.1), size=(0.22, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    neutral = visual.ImageStim(
        win=win,
        name='neutral', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0.1), size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    button = visual.ButtonStim(win, 
        text='Next', font='Arvo',
        pos=(0.5, -0.4),
        letterHeight=0.05,
        size=(0.3, 0.1), borderWidth=0.0,
        fillColor=[-0.4510, 0.0196, 0.4118], borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button',
        depth=-5
    )
    button.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "after_all_audios" ---
    after_all = visual.TextStim(win=win, name='after_all',
        text='You have just listened to all of the stories.\n\nNow you will answer two final questions.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=[0.9137, -0.3647, 0.9765], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    next_4 = visual.ButtonStim(win, 
        text='Next', font='Arvo',
        pos=(0.5, -0.4),
        letterHeight=0.05,
        size=(0.3, 0.1), borderWidth=0.0,
        fillColor=[-0.4510, 0.0196, 0.4118], borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='next_4',
        depth=-1
    )
    next_4.buttonClock = core.Clock()
    mouse_2 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_2.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "favorite_story" ---
    fav = visual.TextStim(win=win, name='fav',
        text='Which story was your favorite?',
        font='Open Sans',
        pos=(0, 0.4), height=0.05, wrapWidth=None, ori=0.0, 
        color=[0.9137, -0.3647, 0.9765], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    fav_story_names = visual.TextStim(win=win, name='fav_story_names',
        text='I Decided To Be Myself And Won a Dance Contest\n\nI Fully Embarrassed Myself In Zoom Class\n\nCamp Lose-a-Friend\n\nFrog Dissection Disaster\n\nThe Birthday Party Prank\n\nLeft Home Alone In a Tornado \n',
        font='Open Sans',
        pos=(0.1, -0.05), height=0.04, wrapWidth=None, ori=0.0, 
        color=[0.9137, -0.3647, 0.9765], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    fav_dance_contest = visual.ImageStim(
        win=win,
        name='fav_dance_contest', 
        image='emojis_grey/dance_contest.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.45, 0.2), size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    fav_zoom_class = visual.ImageStim(
        win=win,
        name='fav_zoom_class', 
        image='emojis_grey/zoom_class.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.45, 0.11), size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    fav_a_friend = visual.ImageStim(
        win=win,
        name='fav_a_friend', 
        image='emojis_grey/a_friend.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.45, 0.02), size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    fav_dissection_disaster = visual.ImageStim(
        win=win,
        name='fav_dissection_disaster', 
        image='emojis_grey/dissection_disaster.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.45, -0.07), size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    fav_party_prank = visual.ImageStim(
        win=win,
        name='fav_party_prank', 
        image='emojis_grey/party_prank.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.45, -0.16), size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-7.0)
    fav_a_tornado = visual.ImageStim(
        win=win,
        name='fav_a_tornado', 
        image='emojis_grey/a_tornado.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.45, -0.25), size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-8.0)
    fav_dance_contest_option = visual.ShapeStim(
        win=win, name='fav_dance_contest_option',
        size=(0.03, 0.03), vertices='circle',
        ori=0.0, pos=(-0.53, 0.2), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-9.0, interpolate=True)
    fav_zoom_class_option = visual.ShapeStim(
        win=win, name='fav_zoom_class_option',
        size=(0.03, 0.03), vertices='circle',
        ori=0.0, pos=(-0.53, 0.11), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-10.0, interpolate=True)
    fav_a_friend_option = visual.ShapeStim(
        win=win, name='fav_a_friend_option',
        size=(0.03, 0.03), vertices='circle',
        ori=0.0, pos=(-0.53, 0.02), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-11.0, interpolate=True)
    fav_dissection_disaster_option = visual.ShapeStim(
        win=win, name='fav_dissection_disaster_option',
        size=(0.03, 0.03), vertices='circle',
        ori=0.0, pos=(-0.53, -0.07), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-12.0, interpolate=True)
    fav_party_prank_option = visual.ShapeStim(
        win=win, name='fav_party_prank_option',
        size=(0.03, 0.03), vertices='circle',
        ori=0.0, pos=(-0.53, -0.16), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-13.0, interpolate=True)
    fav_a_tornado_option = visual.ShapeStim(
        win=win, name='fav_a_tornado_option',
        size=(0.03, 0.03), vertices='circle',
        ori=0.0, pos=(-0.53, -0.25), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-14.0, interpolate=True)
    fav_story_continue = visual.ButtonStim(win, 
        text='Continue', font='Arvo',
        pos=(0.6, -0.4),
        letterHeight=0.05,
        size=(0.3, 0.1), borderWidth=0.0,
        fillColor=[-0.4510, 0.0196, 0.4118], borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='fav_story_continue',
        depth=-15
    )
    fav_story_continue.buttonClock = core.Clock()
    fav_mouse = event.Mouse(win=win)
    x, y = [None, None]
    fav_mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "least_favorite_story" ---
    least_fav = visual.TextStim(win=win, name='least_fav',
        text='Which story was your least favorite?',
        font='Open Sans',
        pos=(0, 0.4), height=0.05, wrapWidth=None, ori=0.0, 
        color=[0.9137, -0.3647, 0.9765], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    least_fav_story_names = visual.TextStim(win=win, name='least_fav_story_names',
        text='I Decided To Be Myself And Won a Dance Contest\n\nI Fully Embarrassed Myself In Zoom Class\n\nCamp Lose-a-Friend\n\nFrog Dissection Disaster\n\nThe Birthday Party Prank\n\nLeft Home Alone In a Tornado \n',
        font='Open Sans',
        pos=(0.1, -0.05), height=0.04, wrapWidth=None, ori=0.0, 
        color=[0.9137, -0.3647, 0.9765], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    least_fav_dance_contest = visual.ImageStim(
        win=win,
        name='least_fav_dance_contest', 
        image='emojis_grey/dance_contest.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.45, 0.2), size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    least_fav_zoom_class = visual.ImageStim(
        win=win,
        name='least_fav_zoom_class', 
        image='emojis_grey/zoom_class.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.45, 0.11), size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    least_fav_a_friend = visual.ImageStim(
        win=win,
        name='least_fav_a_friend', 
        image='emojis_grey/a_friend.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.45, 0.02), size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    least_fav_dissection_disaster = visual.ImageStim(
        win=win,
        name='least_fav_dissection_disaster', 
        image='emojis_grey/dissection_disaster.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.45, -0.07), size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    least_fav_party_prank = visual.ImageStim(
        win=win,
        name='least_fav_party_prank', 
        image='emojis_grey/party_prank.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.45, -0.16), size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-7.0)
    least_fav_a_tornado = visual.ImageStim(
        win=win,
        name='least_fav_a_tornado', 
        image='emojis_grey/a_tornado.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.45, -0.25), size=(0.08, 0.08),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-8.0)
    least_fav_dance_contest_option = visual.ShapeStim(
        win=win, name='least_fav_dance_contest_option',
        size=(0.03, 0.03), vertices='circle',
        ori=0.0, pos=(-0.53, 0.2), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-9.0, interpolate=True)
    least_fav_zoom_class_option = visual.ShapeStim(
        win=win, name='least_fav_zoom_class_option',
        size=(0.03, 0.03), vertices='circle',
        ori=0.0, pos=(-0.53, 0.11), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-10.0, interpolate=True)
    least_fav_a_friend_option = visual.ShapeStim(
        win=win, name='least_fav_a_friend_option',
        size=(0.03, 0.03), vertices='circle',
        ori=0.0, pos=(-0.53, 0.02), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-11.0, interpolate=True)
    least_fav_dissection_disaster_option = visual.ShapeStim(
        win=win, name='least_fav_dissection_disaster_option',
        size=(0.03, 0.03), vertices='circle',
        ori=0.0, pos=(-0.53, -0.07), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-12.0, interpolate=True)
    least_fav_party_prank_option = visual.ShapeStim(
        win=win, name='least_fav_party_prank_option',
        size=(0.03, 0.03), vertices='circle',
        ori=0.0, pos=(-0.53, -0.16), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-13.0, interpolate=True)
    least_fav_a_tornado_option = visual.ShapeStim(
        win=win, name='least_fav_a_tornado_option',
        size=(0.03, 0.03), vertices='circle',
        ori=0.0, pos=(-0.53, -0.25), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-14.0, interpolate=True)
    least_fav_story_continue = visual.ButtonStim(win, 
        text='Continue', font='Arvo',
        pos=(0.6, -0.4),
        letterHeight=0.05,
        size=(0.3, 0.1), borderWidth=0.0,
        fillColor=[-0.4510, 0.0196, 0.4118], borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='least_fav_story_continue',
        depth=-15
    )
    least_fav_story_continue.buttonClock = core.Clock()
    least_fav_mouse = event.Mouse(win=win)
    x, y = [None, None]
    least_fav_mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "check_impedance" ---
    impedance_pause = visual.TextStim(win=win, name='impedance_pause',
        text='Time for a break!',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=[0.9137, -0.3647, 0.9765], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    impedance_end = keyboard.Keyboard(deviceName='impedance_end')
    
    # --- Initialize components for Routine "social_welcome" ---
    text_3 = visual.TextStim(win=win, name='text_3',
        text='Welcome to the social task',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=[0.9137, -0.3647, 0.9765], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    social_start_button = visual.ButtonStim(win, 
        text='Start', font='Arvo',
        pos=(0.5, -0.4),
        letterHeight=0.05,
        size=(0.3, 0.1), borderWidth=0.0,
        fillColor=[-0.4510, 0.0196, 0.4118], borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='social_start_button',
        depth=-1
    )
    social_start_button.buttonClock = core.Clock()
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "social_starts" ---
    countdown = visual.TextStim(win=win, name='countdown',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    social_end_sound = sound.Sound(
        'audio_files/normalized/new_beep_sound.wav', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='social_end_sound',    name='social_end_sound'
    )
    social_end_sound.setVolume(0.8)
    
    # --- Initialize components for Routine "end_experiment" ---
    thank_you = visual.TextStim(win=win, name='thank_you',
        text='Thank you!',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=[0.9137, -0.3647, 0.9765], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "setup" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('setup.started', globalClock.getTime(format='float'))
    start_task.keys = []
    start_task.rt = []
    _start_task_allKeys = []
    # keep track of which components have finished
    setupComponents = [text, start_task]
    for thisComponent in setupComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "setup" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # *start_task* updates
        waitOnFlip = False
        
        # if start_task is starting this frame...
        if start_task.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start_task.frameNStart = frameN  # exact frame index
            start_task.tStart = t  # local t and not account for scr refresh
            start_task.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start_task, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start_task.started')
            # update status
            start_task.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(start_task.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(start_task.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if start_task.status == STARTED and not waitOnFlip:
            theseKeys = start_task.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _start_task_allKeys.extend(theseKeys)
            if len(_start_task_allKeys):
                start_task.keys = _start_task_allKeys[-1].name  # just the last key pressed
                start_task.rt = _start_task_allKeys[-1].rt
                start_task.duration = _start_task_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in setupComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "setup" ---
    for thisComponent in setupComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('setup.stopped', globalClock.getTime(format='float'))
    # check responses
    if start_task.keys in ['', [], None]:  # No response was made
        start_task.keys = None
    thisExp.addData('start_task.keys',start_task.keys)
    if start_task.keys != None:  # we had a response
        thisExp.addData('start_task.rt', start_task.rt)
        thisExp.addData('start_task.duration', start_task.duration)
    thisExp.nextEntry()
    # the Routine "setup" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "resting_state_welcome" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('resting_state_welcome.started', globalClock.getTime(format='float'))
    rst_key_start.keys = []
    rst_key_start.rt = []
    _rst_key_start_allKeys = []
    # Run 'Begin Routine' code from protocol_starts
    import time
    from pylsl import local_clock
    
    win.mouseVisible = False
    marker_time = time.time()
    
    outlet.push_sample([200])
    outlet.push_sample([marker_time])
    # keep track of which components have finished
    resting_state_welcomeComponents = [rst_welcome, rst_key_start]
    for thisComponent in resting_state_welcomeComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "resting_state_welcome" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *rst_welcome* updates
        
        # if rst_welcome is starting this frame...
        if rst_welcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            rst_welcome.frameNStart = frameN  # exact frame index
            rst_welcome.tStart = t  # local t and not account for scr refresh
            rst_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rst_welcome, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rst_welcome.started')
            # update status
            rst_welcome.status = STARTED
            rst_welcome.setAutoDraw(True)
        
        # if rst_welcome is active this frame...
        if rst_welcome.status == STARTED:
            # update params
            pass
        
        # *rst_key_start* updates
        waitOnFlip = False
        
        # if rst_key_start is starting this frame...
        if rst_key_start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            rst_key_start.frameNStart = frameN  # exact frame index
            rst_key_start.tStart = t  # local t and not account for scr refresh
            rst_key_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rst_key_start, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rst_key_start.started')
            # update status
            rst_key_start.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(rst_key_start.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(rst_key_start.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if rst_key_start.status == STARTED and not waitOnFlip:
            theseKeys = rst_key_start.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _rst_key_start_allKeys.extend(theseKeys)
            if len(_rst_key_start_allKeys):
                rst_key_start.keys = _rst_key_start_allKeys[-1].name  # just the last key pressed
                rst_key_start.rt = _rst_key_start_allKeys[-1].rt
                rst_key_start.duration = _rst_key_start_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in resting_state_welcomeComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "resting_state_welcome" ---
    for thisComponent in resting_state_welcomeComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('resting_state_welcome.stopped', globalClock.getTime(format='float'))
    # check responses
    if rst_key_start.keys in ['', [], None]:  # No response was made
        rst_key_start.keys = None
    thisExp.addData('rst_key_start.keys',rst_key_start.keys)
    if rst_key_start.keys != None:  # we had a response
        thisExp.addData('rst_key_start.rt', rst_key_start.rt)
        thisExp.addData('rst_key_start.duration', rst_key_start.duration)
    thisExp.nextEntry()
    # the Routine "resting_state_welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "resting_state" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('resting_state.started', globalClock.getTime(format='float'))
    # Run 'Begin Routine' code from rst_lsl
    import time
    from pylsl import local_clock
    import socket
    
    
    
    win.mouseVisible = False
    marker_time = time.time()
    outlet.push_sample([10])
    outlet.push_sample([marker_time])
    
    # keep track of which components have finished
    resting_stateComponents = [rst_screen]
    for thisComponent in resting_stateComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "resting_state" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 300.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *rst_screen* updates
        
        # if rst_screen is starting this frame...
        if rst_screen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            rst_screen.frameNStart = frameN  # exact frame index
            rst_screen.tStart = t  # local t and not account for scr refresh
            rst_screen.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rst_screen, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rst_screen.started')
            # update status
            rst_screen.status = STARTED
            rst_screen.setAutoDraw(True)
        
        # if rst_screen is active this frame...
        if rst_screen.status == STARTED:
            # update params
            pass
        
        # if rst_screen is stopping this frame...
        if rst_screen.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > rst_screen.tStartRefresh + 300-frameTolerance:
                # keep track of stop time/frame for later
                rst_screen.tStop = t  # not accounting for scr refresh
                rst_screen.tStopRefresh = tThisFlipGlobal  # on global time
                rst_screen.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rst_screen.stopped')
                # update status
                rst_screen.status = FINISHED
                rst_screen.setAutoDraw(False)
        # Run 'Each Frame' code from rst_lsl
        keys = event.getKeys(keyList=['left', 'right'])
        
        if 'right' in keys:
            continueRoutine = False
            
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in resting_stateComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "resting_state" ---
    for thisComponent in resting_stateComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('resting_state.stopped', globalClock.getTime(format='float'))
    # Run 'End Routine' code from rst_lsl
    import time
    from pylsl import local_clock
    
    win.mouseVisible = True
    marker_time = time.time()
    
    outlet.push_sample([11])
    outlet.push_sample([marker_time])
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-300.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "audio_welcome" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('audio_welcome.started', globalClock.getTime(format='float'))
    audio_next.keys = []
    audio_next.rt = []
    _audio_next_allKeys = []
    # Run 'Begin Routine' code from start_audio_task
    import time
    from pylsl import local_clock
    
    win.mouseVisible = False
    marker_time = time.time()
    
    outlet.push_sample([500])
    outlet.push_sample([marker_time])
    # keep track of which components have finished
    audio_welcomeComponents = [welcome_instr, audio_next]
    for thisComponent in audio_welcomeComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "audio_welcome" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welcome_instr* updates
        
        # if welcome_instr is starting this frame...
        if welcome_instr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcome_instr.frameNStart = frameN  # exact frame index
            welcome_instr.tStart = t  # local t and not account for scr refresh
            welcome_instr.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcome_instr, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcome_instr.started')
            # update status
            welcome_instr.status = STARTED
            welcome_instr.setAutoDraw(True)
        
        # if welcome_instr is active this frame...
        if welcome_instr.status == STARTED:
            # update params
            pass
        
        # *audio_next* updates
        waitOnFlip = False
        
        # if audio_next is starting this frame...
        if audio_next.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            audio_next.frameNStart = frameN  # exact frame index
            audio_next.tStart = t  # local t and not account for scr refresh
            audio_next.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(audio_next, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'audio_next.started')
            # update status
            audio_next.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(audio_next.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(audio_next.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if audio_next.status == STARTED and not waitOnFlip:
            theseKeys = audio_next.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _audio_next_allKeys.extend(theseKeys)
            if len(_audio_next_allKeys):
                audio_next.keys = _audio_next_allKeys[-1].name  # just the last key pressed
                audio_next.rt = _audio_next_allKeys[-1].rt
                audio_next.duration = _audio_next_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in audio_welcomeComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "audio_welcome" ---
    for thisComponent in audio_welcomeComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('audio_welcome.stopped', globalClock.getTime(format='float'))
    # check responses
    if audio_next.keys in ['', [], None]:  # No response was made
        audio_next.keys = None
    thisExp.addData('audio_next.keys',audio_next.keys)
    if audio_next.keys != None:  # we had a response
        thisExp.addData('audio_next.rt', audio_next.rt)
        thisExp.addData('audio_next.duration', audio_next.duration)
    thisExp.nextEntry()
    # the Routine "audio_welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "relax" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('relax.started', globalClock.getTime(format='float'))
    audio_next2.keys = []
    audio_next2.rt = []
    _audio_next2_allKeys = []
    # Run 'Begin Routine' code from code_3
    import socket
    
    
    
    server_ip = "10.10.10.45"
    server_port = 4444
    
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    
    
    try: 
        client_socket.connect((server_ip, server_port))
        print(f'connect to server at {server_ip}: {server_port}')
        trigger = "Mic_trigger"
        client_socket.sendall(trigger.encode('utf-8'))
        print('trigger sent!')
    finally:
        client_socket.close()
    # keep track of which components have finished
    relaxComponents = [text_2, audio_next2]
    for thisComponent in relaxComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "relax" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_2* updates
        
        # if text_2 is starting this frame...
        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_2.started')
            # update status
            text_2.status = STARTED
            text_2.setAutoDraw(True)
        
        # if text_2 is active this frame...
        if text_2.status == STARTED:
            # update params
            pass
        
        # *audio_next2* updates
        waitOnFlip = False
        
        # if audio_next2 is starting this frame...
        if audio_next2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            audio_next2.frameNStart = frameN  # exact frame index
            audio_next2.tStart = t  # local t and not account for scr refresh
            audio_next2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(audio_next2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'audio_next2.started')
            # update status
            audio_next2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(audio_next2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(audio_next2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if audio_next2.status == STARTED and not waitOnFlip:
            theseKeys = audio_next2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _audio_next2_allKeys.extend(theseKeys)
            if len(_audio_next2_allKeys):
                audio_next2.keys = _audio_next2_allKeys[-1].name  # just the last key pressed
                audio_next2.rt = _audio_next2_allKeys[-1].rt
                audio_next2.duration = _audio_next2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in relaxComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "relax" ---
    for thisComponent in relaxComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('relax.stopped', globalClock.getTime(format='float'))
    # check responses
    if audio_next2.keys in ['', [], None]:  # No response was made
        audio_next2.keys = None
    thisExp.addData('audio_next2.keys',audio_next2.keys)
    if audio_next2.keys != None:  # we had a response
        thisExp.addData('audio_next2.rt', audio_next2.rt)
        thisExp.addData('audio_next2.duration', audio_next2.duration)
    thisExp.nextEntry()
    # the Routine "relax" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    conditions = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('audio_conditions.xlsx'),
        seed=None, name='conditions')
    thisExp.addLoop(conditions)  # add the loop to the experiment
    thisCondition = conditions.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisCondition.rgb)
    if thisCondition != None:
        for paramName in thisCondition:
            globals()[paramName] = thisCondition[paramName]
    
    for thisCondition in conditions:
        currentLoop = conditions
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisCondition.rgb)
        if thisCondition != None:
            for paramName in thisCondition:
                globals()[paramName] = thisCondition[paramName]
        
        # --- Prepare to start Routine "audio_instructions" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('audio_instructions.started', globalClock.getTime(format='float'))
        audio_name.setText(AudioName)
        emoji.setImage(EmojiFile)
        audio_next3.keys = []
        audio_next3.rt = []
        _audio_next3_allKeys = []
        # keep track of which components have finished
        audio_instructionsComponents = [disp_audio_name, audio_name, emoji, audio_next3]
        for thisComponent in audio_instructionsComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "audio_instructions" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *disp_audio_name* updates
            
            # if disp_audio_name is starting this frame...
            if disp_audio_name.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                disp_audio_name.frameNStart = frameN  # exact frame index
                disp_audio_name.tStart = t  # local t and not account for scr refresh
                disp_audio_name.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(disp_audio_name, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'disp_audio_name.started')
                # update status
                disp_audio_name.status = STARTED
                disp_audio_name.setAutoDraw(True)
            
            # if disp_audio_name is active this frame...
            if disp_audio_name.status == STARTED:
                # update params
                pass
            
            # *audio_name* updates
            
            # if audio_name is starting this frame...
            if audio_name.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                audio_name.frameNStart = frameN  # exact frame index
                audio_name.tStart = t  # local t and not account for scr refresh
                audio_name.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(audio_name, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'audio_name.started')
                # update status
                audio_name.status = STARTED
                audio_name.setAutoDraw(True)
            
            # if audio_name is active this frame...
            if audio_name.status == STARTED:
                # update params
                pass
            
            # *emoji* updates
            
            # if emoji is starting this frame...
            if emoji.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                emoji.frameNStart = frameN  # exact frame index
                emoji.tStart = t  # local t and not account for scr refresh
                emoji.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(emoji, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'emoji.started')
                # update status
                emoji.status = STARTED
                emoji.setAutoDraw(True)
            
            # if emoji is active this frame...
            if emoji.status == STARTED:
                # update params
                pass
            
            # *audio_next3* updates
            waitOnFlip = False
            
            # if audio_next3 is starting this frame...
            if audio_next3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                audio_next3.frameNStart = frameN  # exact frame index
                audio_next3.tStart = t  # local t and not account for scr refresh
                audio_next3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(audio_next3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'audio_next3.started')
                # update status
                audio_next3.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(audio_next3.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(audio_next3.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if audio_next3.status == STARTED and not waitOnFlip:
                theseKeys = audio_next3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _audio_next3_allKeys.extend(theseKeys)
                if len(_audio_next3_allKeys):
                    audio_next3.keys = _audio_next3_allKeys[-1].name  # just the last key pressed
                    audio_next3.rt = _audio_next3_allKeys[-1].rt
                    audio_next3.duration = _audio_next3_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in audio_instructionsComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "audio_instructions" ---
        for thisComponent in audio_instructionsComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('audio_instructions.stopped', globalClock.getTime(format='float'))
        # check responses
        if audio_next3.keys in ['', [], None]:  # No response was made
            audio_next3.keys = None
        conditions.addData('audio_next3.keys',audio_next3.keys)
        if audio_next3.keys != None:  # we had a response
            conditions.addData('audio_next3.rt', audio_next3.rt)
            conditions.addData('audio_next3.duration', audio_next3.duration)
        # the Routine "audio_instructions" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "audio_rest" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('audio_rest.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from rest_code
        import time
        from pylsl import local_clock
        
        win.mouseVisible = False
        marker_time = time.time()
        
        outlet.push_sample([100])
        outlet.push_sample([marker_time])
        # keep track of which components have finished
        audio_restComponents = [rest_before_audio]
        for thisComponent in audio_restComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "audio_rest" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 10.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *rest_before_audio* updates
            
            # if rest_before_audio is starting this frame...
            if rest_before_audio.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rest_before_audio.frameNStart = frameN  # exact frame index
                rest_before_audio.tStart = t  # local t and not account for scr refresh
                rest_before_audio.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rest_before_audio, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rest_before_audio.started')
                # update status
                rest_before_audio.status = STARTED
                rest_before_audio.setAutoDraw(True)
            
            # if rest_before_audio is active this frame...
            if rest_before_audio.status == STARTED:
                # update params
                pass
            
            # if rest_before_audio is stopping this frame...
            if rest_before_audio.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rest_before_audio.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    rest_before_audio.tStop = t  # not accounting for scr refresh
                    rest_before_audio.tStopRefresh = tThisFlipGlobal  # on global time
                    rest_before_audio.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rest_before_audio.stopped')
                    # update status
                    rest_before_audio.status = FINISHED
                    rest_before_audio.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in audio_restComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "audio_rest" ---
        for thisComponent in audio_restComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('audio_rest.stopped', globalClock.getTime(format='float'))
        # Run 'End Routine' code from rest_code
        import time
        from pylsl import local_clock
        
        win.mouseVisible = False
        marker_time = time.time()
        
        outlet.push_sample([101])
        outlet.push_sample([marker_time])
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-10.000000)
        
        # --- Prepare to start Routine "audio" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('audio.started', globalClock.getTime(format='float'))
        audio_file.setSound(AudioFile, hamming=True)
        audio_file.setVolume(1.0, log=False)
        audio_file.seek(0)
        # Run 'Begin Routine' code from send_triggers
        import time
        from pylsl import local_clock
        
        win.mouseVisible = False
        marker_time = time.time()
        
        if AudioFile == 'audio_files/normalized/Camp_Lose_A_Friend.wav':
            outlet.push_sample([20])
            outlet.push_sample([marker_time])
        elif AudioFile == 'audio_files/normalized/Frog_Dissection_Disaster.wav':
            outlet.push_sample([30])
            outlet.push_sample([marker_time])
        elif AudioFile == 'audio_files/normalized/I_Decided_To_Be_Myself_And_Won_A_Dance_Contest.wav':
            outlet.push_sample([40])
            outlet.push_sample([marker_time])
        elif AudioFile == 'audio_files/normalized/I_Fully_Embarrassed_Myself_In_Zoom_Class1.wav':
            outlet.push_sample([50])
            outlet.push_sample([marker_time])
        elif AudioFile == 'audio_files/normalized/Left_Home_Alone_in_a_Tornado.wav':
            outlet.push_sample([60])
            outlet.push_sample([marker_time])
        elif AudioFile == 'audio_files/normalized/The_Birthday_Party_Prank.wav':
            outlet.push_sample([70])
            outlet.push_sample([marker_time])
        # keep track of which components have finished
        audioComponents = [audio_crosshair, audio_file]
        for thisComponent in audioComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "audio" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *audio_crosshair* updates
            
            # if audio_crosshair is starting this frame...
            if audio_crosshair.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                audio_crosshair.frameNStart = frameN  # exact frame index
                audio_crosshair.tStart = t  # local t and not account for scr refresh
                audio_crosshair.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(audio_crosshair, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'audio_crosshair.started')
                # update status
                audio_crosshair.status = STARTED
                audio_crosshair.setAutoDraw(True)
            
            # if audio_crosshair is active this frame...
            if audio_crosshair.status == STARTED:
                # update params
                pass
            
            # if audio_file is starting this frame...
            if audio_file.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                audio_file.frameNStart = frameN  # exact frame index
                audio_file.tStart = t  # local t and not account for scr refresh
                audio_file.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('audio_file.started', tThisFlipGlobal)
                # update status
                audio_file.status = STARTED
                audio_file.play(when=win)  # sync with win flip
            # update audio_file status according to whether it's playing
            if audio_file.isPlaying:
                audio_file.status = STARTED
            elif audio_file.isFinished:
                audio_file.status = FINISHED
            # Run 'Each Frame' code from send_triggers
            if audio_file.status == FINISHED:
                continueRoutine = False
                
            keys = event.getKeys(keyList=['left', 'right'])
            
            if 'right' in keys:
                continueRoutine = False
                
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in audioComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "audio" ---
        for thisComponent in audioComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('audio.stopped', globalClock.getTime(format='float'))
        audio_file.pause()  # ensure sound has stopped at end of Routine
        # Run 'End Routine' code from send_triggers
        import time
        from pylsl import local_clock
        
        win.mouseVisible = True
        marker_time = time.time()
        
        if AudioFile == 'audio_files/normalized/Camp_Lose_A_Friend.wav':
            outlet.push_sample([21])
            outlet.push_sample([marker_time])
        elif AudioFile == 'audio_files/normalized/Frog_Dissection_Disaster.wav':
            outlet.push_sample([31])
            outlet.push_sample([marker_time])
        elif AudioFile == 'audio_files/normalized/I_Decided_To_Be_Myself_And_Won_A_Dance_Contest.wav':
            outlet.push_sample([41])
            outlet.push_sample([marker_time])
        elif AudioFile == 'audio_files/normalized/I_Fully_Embarrassed_Myself_In_Zoom_Class1.wav':
            outlet.push_sample([51])
            outlet.push_sample([marker_time])
        elif AudioFile == 'audio_files/normalized/Left_Home_Alone_in_a_Tornado.wav':
            outlet.push_sample([61])
            outlet.push_sample([marker_time])
        elif AudioFile == 'audio_files/normalized/The_Birthday_Party_Prank.wav':
            outlet.push_sample([71])
            outlet.push_sample([marker_time])
        
        # the Routine "audio" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "after_audio_instructions" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('after_audio_instructions.started', globalClock.getTime(format='float'))
        audio_name2.setText(AudioName)
        emoji2.setImage(EmojiFile)
        # reset next_3 to account for continued clicks & clear times on/off
        next_3.reset()
        # keep track of which components have finished
        after_audio_instructionsComponents = [inst_after, audio_name2, emoji2, inst_after2, next_3]
        for thisComponent in after_audio_instructionsComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "after_audio_instructions" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *inst_after* updates
            
            # if inst_after is starting this frame...
            if inst_after.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                inst_after.frameNStart = frameN  # exact frame index
                inst_after.tStart = t  # local t and not account for scr refresh
                inst_after.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(inst_after, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'inst_after.started')
                # update status
                inst_after.status = STARTED
                inst_after.setAutoDraw(True)
            
            # if inst_after is active this frame...
            if inst_after.status == STARTED:
                # update params
                pass
            
            # *audio_name2* updates
            
            # if audio_name2 is starting this frame...
            if audio_name2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                audio_name2.frameNStart = frameN  # exact frame index
                audio_name2.tStart = t  # local t and not account for scr refresh
                audio_name2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(audio_name2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'audio_name2.started')
                # update status
                audio_name2.status = STARTED
                audio_name2.setAutoDraw(True)
            
            # if audio_name2 is active this frame...
            if audio_name2.status == STARTED:
                # update params
                pass
            
            # *emoji2* updates
            
            # if emoji2 is starting this frame...
            if emoji2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                emoji2.frameNStart = frameN  # exact frame index
                emoji2.tStart = t  # local t and not account for scr refresh
                emoji2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(emoji2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'emoji2.started')
                # update status
                emoji2.status = STARTED
                emoji2.setAutoDraw(True)
            
            # if emoji2 is active this frame...
            if emoji2.status == STARTED:
                # update params
                pass
            
            # *inst_after2* updates
            
            # if inst_after2 is starting this frame...
            if inst_after2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                inst_after2.frameNStart = frameN  # exact frame index
                inst_after2.tStart = t  # local t and not account for scr refresh
                inst_after2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(inst_after2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'inst_after2.started')
                # update status
                inst_after2.status = STARTED
                inst_after2.setAutoDraw(True)
            
            # if inst_after2 is active this frame...
            if inst_after2.status == STARTED:
                # update params
                pass
            # *next_3* updates
            
            # if next_3 is starting this frame...
            if next_3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                next_3.frameNStart = frameN  # exact frame index
                next_3.tStart = t  # local t and not account for scr refresh
                next_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(next_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'next_3.started')
                # update status
                next_3.status = STARTED
                next_3.setAutoDraw(True)
            
            # if next_3 is active this frame...
            if next_3.status == STARTED:
                # update params
                pass
                # check whether next_3 has been pressed
                if next_3.isClicked:
                    if not next_3.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        next_3.timesOn.append(next_3.buttonClock.getTime())
                        next_3.timesOff.append(next_3.buttonClock.getTime())
                    elif len(next_3.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        next_3.timesOff[-1] = next_3.buttonClock.getTime()
                    if not next_3.wasClicked:
                        # end routine when next_3 is clicked
                        continueRoutine = False
                    if not next_3.wasClicked:
                        # run callback code when next_3 is clicked
                        pass
            # take note of whether next_3 was clicked, so that next frame we know if clicks are new
            next_3.wasClicked = next_3.isClicked and next_3.status == STARTED
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in after_audio_instructionsComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "after_audio_instructions" ---
        for thisComponent in after_audio_instructionsComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('after_audio_instructions.stopped', globalClock.getTime(format='float'))
        conditions.addData('next_3.numClicks', next_3.numClicks)
        if next_3.numClicks:
           conditions.addData('next_3.timesOn', next_3.timesOn)
           conditions.addData('next_3.timesOff', next_3.timesOff)
        else:
           conditions.addData('next_3.timesOn', "")
           conditions.addData('next_3.timesOff', "")
        # the Routine "after_audio_instructions" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        questions_loop = data.TrialHandler(nReps=1.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('questions.xlsx'),
            seed=None, name='questions_loop')
        thisExp.addLoop(questions_loop)  # add the loop to the experiment
        thisQuestions_loop = questions_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisQuestions_loop.rgb)
        if thisQuestions_loop != None:
            for paramName in thisQuestions_loop:
                globals()[paramName] = thisQuestions_loop[paramName]
        
        for thisQuestions_loop in questions_loop:
            currentLoop = questions_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisQuestions_loop.rgb)
            if thisQuestions_loop != None:
                for paramName in thisQuestions_loop:
                    globals()[paramName] = thisQuestions_loop[paramName]
            
            # --- Prepare to start Routine "questions" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('questions.started', globalClock.getTime(format='float'))
            question.setText(Question)
            slider.reset()
            disagree.setImage(DisagreeEmoji)
            agree.setImage(AgreeEmoji)
            neutral.setImage(NeutralEmoji)
            # reset button to account for continued clicks & clear times on/off
            button.reset()
            # Run 'Begin Routine' code from answer_question_lsl
            import time
            from pylsl import local_clock
            
            win.mouseVisible = True
            marker_time = time.time()
            
            outlet.push_sample([300])
            outlet.push_sample([marker_time])
            # keep track of which components have finished
            questionsComponents = [question, slider, disagree, agree, neutral, button]
            for thisComponent in questionsComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "questions" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *question* updates
                
                # if question is starting this frame...
                if question.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    question.frameNStart = frameN  # exact frame index
                    question.tStart = t  # local t and not account for scr refresh
                    question.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(question, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'question.started')
                    # update status
                    question.status = STARTED
                    question.setAutoDraw(True)
                
                # if question is active this frame...
                if question.status == STARTED:
                    # update params
                    pass
                
                # *slider* updates
                
                # if slider is starting this frame...
                if slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    slider.frameNStart = frameN  # exact frame index
                    slider.tStart = t  # local t and not account for scr refresh
                    slider.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(slider, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'slider.started')
                    # update status
                    slider.status = STARTED
                    slider.setAutoDraw(True)
                
                # if slider is active this frame...
                if slider.status == STARTED:
                    # update params
                    pass
                
                # *disagree* updates
                
                # if disagree is starting this frame...
                if disagree.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    disagree.frameNStart = frameN  # exact frame index
                    disagree.tStart = t  # local t and not account for scr refresh
                    disagree.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(disagree, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'disagree.started')
                    # update status
                    disagree.status = STARTED
                    disagree.setAutoDraw(True)
                
                # if disagree is active this frame...
                if disagree.status == STARTED:
                    # update params
                    pass
                
                # *agree* updates
                
                # if agree is starting this frame...
                if agree.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    agree.frameNStart = frameN  # exact frame index
                    agree.tStart = t  # local t and not account for scr refresh
                    agree.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(agree, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'agree.started')
                    # update status
                    agree.status = STARTED
                    agree.setAutoDraw(True)
                
                # if agree is active this frame...
                if agree.status == STARTED:
                    # update params
                    pass
                
                # *neutral* updates
                
                # if neutral is starting this frame...
                if neutral.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    neutral.frameNStart = frameN  # exact frame index
                    neutral.tStart = t  # local t and not account for scr refresh
                    neutral.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(neutral, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'neutral.started')
                    # update status
                    neutral.status = STARTED
                    neutral.setAutoDraw(True)
                
                # if neutral is active this frame...
                if neutral.status == STARTED:
                    # update params
                    pass
                # *button* updates
                
                # if button is starting this frame...
                if button.status == NOT_STARTED and slider.rating:
                    # keep track of start time/frame for later
                    button.frameNStart = frameN  # exact frame index
                    button.tStart = t  # local t and not account for scr refresh
                    button.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(button, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'button.started')
                    # update status
                    button.status = STARTED
                    button.setAutoDraw(True)
                
                # if button is active this frame...
                if button.status == STARTED:
                    # update params
                    pass
                    # check whether button has been pressed
                    if button.isClicked:
                        if not button.wasClicked:
                            # if this is a new click, store time of first click and clicked until
                            button.timesOn.append(button.buttonClock.getTime())
                            button.timesOff.append(button.buttonClock.getTime())
                        elif len(button.timesOff):
                            # if click is continuing from last frame, update time of clicked until
                            button.timesOff[-1] = button.buttonClock.getTime()
                        if not button.wasClicked:
                            # end routine when button is clicked
                            continueRoutine = False
                        if not button.wasClicked:
                            # run callback code when button is clicked
                            pass
                # take note of whether button was clicked, so that next frame we know if clicks are new
                button.wasClicked = button.isClicked and button.status == STARTED
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in questionsComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "questions" ---
            for thisComponent in questionsComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('questions.stopped', globalClock.getTime(format='float'))
            questions_loop.addData('slider.response', slider.getRating())
            questions_loop.addData('slider.rt', slider.getRT())
            questions_loop.addData('button.numClicks', button.numClicks)
            if button.numClicks:
               questions_loop.addData('button.timesOn', button.timesOn)
               questions_loop.addData('button.timesOff', button.timesOff)
            else:
               questions_loop.addData('button.timesOn', "")
               questions_loop.addData('button.timesOff', "")
            # Run 'End Routine' code from answer_question_lsl
            import time
            from pylsl import local_clock
            
            win.mouseVisible = False
            marker_time = time.time()
            
            outlet.push_sample([301])
            outlet.push_sample([slider.rating])
            outlet.push_sample([marker_time])
            
            # the Routine "questions" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'questions_loop'
        
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'conditions'
    
    
    # --- Prepare to start Routine "after_all_audios" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('after_all_audios.started', globalClock.getTime(format='float'))
    # reset next_4 to account for continued clicks & clear times on/off
    next_4.reset()
    # setup some python lists for storing info about the mouse_2
    mouse_2.x = []
    mouse_2.y = []
    mouse_2.leftButton = []
    mouse_2.midButton = []
    mouse_2.rightButton = []
    mouse_2.time = []
    mouse_2.clicked_name = []
    gotValidClick = False  # until a click is received
    # Run 'Begin Routine' code from code_2
    win.mouseVisible = True
    # keep track of which components have finished
    after_all_audiosComponents = [after_all, next_4, mouse_2]
    for thisComponent in after_all_audiosComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "after_all_audios" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *after_all* updates
        
        # if after_all is starting this frame...
        if after_all.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            after_all.frameNStart = frameN  # exact frame index
            after_all.tStart = t  # local t and not account for scr refresh
            after_all.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(after_all, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'after_all.started')
            # update status
            after_all.status = STARTED
            after_all.setAutoDraw(True)
        
        # if after_all is active this frame...
        if after_all.status == STARTED:
            # update params
            pass
        # *next_4* updates
        
        # if next_4 is starting this frame...
        if next_4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            next_4.frameNStart = frameN  # exact frame index
            next_4.tStart = t  # local t and not account for scr refresh
            next_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(next_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'next_4.started')
            # update status
            next_4.status = STARTED
            next_4.setAutoDraw(True)
        
        # if next_4 is active this frame...
        if next_4.status == STARTED:
            # update params
            pass
            # check whether next_4 has been pressed
            if next_4.isClicked:
                if not next_4.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    next_4.timesOn.append(next_4.buttonClock.getTime())
                    next_4.timesOff.append(next_4.buttonClock.getTime())
                elif len(next_4.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    next_4.timesOff[-1] = next_4.buttonClock.getTime()
                if not next_4.wasClicked:
                    # end routine when next_4 is clicked
                    continueRoutine = False
                if not next_4.wasClicked:
                    # run callback code when next_4 is clicked
                    pass
        # take note of whether next_4 was clicked, so that next frame we know if clicks are new
        next_4.wasClicked = next_4.isClicked and next_4.status == STARTED
        # *mouse_2* updates
        
        # if mouse_2 is starting this frame...
        if mouse_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_2.frameNStart = frameN  # exact frame index
            mouse_2.tStart = t  # local t and not account for scr refresh
            mouse_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('mouse_2.started', t)
            # update status
            mouse_2.status = STARTED
            mouse_2.mouseClock.reset()
            prevButtonState = mouse_2.getPressed()  # if button is down already this ISN'T a new click
        
        # if mouse_2 is stopping this frame...
        if mouse_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > mouse_2.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                mouse_2.tStop = t  # not accounting for scr refresh
                mouse_2.tStopRefresh = tThisFlipGlobal  # on global time
                mouse_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.addData('mouse_2.stopped', t)
                # update status
                mouse_2.status = FINISHED
        if mouse_2.status == STARTED:  # only update if started and not finished!
            buttons = mouse_2.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(next_4, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_2):
                            gotValidClick = True
                            mouse_2.clicked_name.append(obj.name)
                    x, y = mouse_2.getPos()
                    mouse_2.x.append(x)
                    mouse_2.y.append(y)
                    buttons = mouse_2.getPressed()
                    mouse_2.leftButton.append(buttons[0])
                    mouse_2.midButton.append(buttons[1])
                    mouse_2.rightButton.append(buttons[2])
                    mouse_2.time.append(mouse_2.mouseClock.getTime())
                    
                    continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in after_all_audiosComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "after_all_audios" ---
    for thisComponent in after_all_audiosComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('after_all_audios.stopped', globalClock.getTime(format='float'))
    thisExp.addData('next_4.numClicks', next_4.numClicks)
    if next_4.numClicks:
       thisExp.addData('next_4.timesOn', next_4.timesOn)
       thisExp.addData('next_4.timesOff', next_4.timesOff)
    else:
       thisExp.addData('next_4.timesOn', "")
       thisExp.addData('next_4.timesOff', "")
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_2.x', mouse_2.x)
    thisExp.addData('mouse_2.y', mouse_2.y)
    thisExp.addData('mouse_2.leftButton', mouse_2.leftButton)
    thisExp.addData('mouse_2.midButton', mouse_2.midButton)
    thisExp.addData('mouse_2.rightButton', mouse_2.rightButton)
    thisExp.addData('mouse_2.time', mouse_2.time)
    thisExp.addData('mouse_2.clicked_name', mouse_2.clicked_name)
    thisExp.nextEntry()
    # the Routine "after_all_audios" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "favorite_story" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('favorite_story.started', globalClock.getTime(format='float'))
    # Run 'Begin Routine' code from align_text
    fav_story_names.alignText = 'left'
    # reset fav_story_continue to account for continued clicks & clear times on/off
    fav_story_continue.reset()
    # setup some python lists for storing info about the fav_mouse
    fav_mouse.x = []
    fav_mouse.y = []
    fav_mouse.leftButton = []
    fav_mouse.midButton = []
    fav_mouse.rightButton = []
    fav_mouse.time = []
    fav_mouse.clicked_name = []
    gotValidClick = False  # until a click is received
    # Run 'Begin Routine' code from fav_story_code
    import time
    from pylsl import local_clock
    
    win.mouseVisible = True
    marker_time = time.time()
    
    most_favorite_story = None
    options = [fav_dance_contest_option, fav_zoom_class_option, fav_a_friend_option, fav_dissection_disaster_option, fav_party_prank_option, fav_a_tornado_option]
    
    outlet.push_sample([302])
    outlet.push_sample([marker_time])
    # keep track of which components have finished
    favorite_storyComponents = [fav, fav_story_names, fav_dance_contest, fav_zoom_class, fav_a_friend, fav_dissection_disaster, fav_party_prank, fav_a_tornado, fav_dance_contest_option, fav_zoom_class_option, fav_a_friend_option, fav_dissection_disaster_option, fav_party_prank_option, fav_a_tornado_option, fav_story_continue, fav_mouse]
    for thisComponent in favorite_storyComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "favorite_story" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fav* updates
        
        # if fav is starting this frame...
        if fav.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fav.frameNStart = frameN  # exact frame index
            fav.tStart = t  # local t and not account for scr refresh
            fav.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fav, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fav.started')
            # update status
            fav.status = STARTED
            fav.setAutoDraw(True)
        
        # if fav is active this frame...
        if fav.status == STARTED:
            # update params
            pass
        
        # *fav_story_names* updates
        
        # if fav_story_names is starting this frame...
        if fav_story_names.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fav_story_names.frameNStart = frameN  # exact frame index
            fav_story_names.tStart = t  # local t and not account for scr refresh
            fav_story_names.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fav_story_names, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fav_story_names.started')
            # update status
            fav_story_names.status = STARTED
            fav_story_names.setAutoDraw(True)
        
        # if fav_story_names is active this frame...
        if fav_story_names.status == STARTED:
            # update params
            pass
        
        # *fav_dance_contest* updates
        
        # if fav_dance_contest is starting this frame...
        if fav_dance_contest.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fav_dance_contest.frameNStart = frameN  # exact frame index
            fav_dance_contest.tStart = t  # local t and not account for scr refresh
            fav_dance_contest.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fav_dance_contest, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fav_dance_contest.started')
            # update status
            fav_dance_contest.status = STARTED
            fav_dance_contest.setAutoDraw(True)
        
        # if fav_dance_contest is active this frame...
        if fav_dance_contest.status == STARTED:
            # update params
            pass
        
        # *fav_zoom_class* updates
        
        # if fav_zoom_class is starting this frame...
        if fav_zoom_class.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fav_zoom_class.frameNStart = frameN  # exact frame index
            fav_zoom_class.tStart = t  # local t and not account for scr refresh
            fav_zoom_class.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fav_zoom_class, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fav_zoom_class.started')
            # update status
            fav_zoom_class.status = STARTED
            fav_zoom_class.setAutoDraw(True)
        
        # if fav_zoom_class is active this frame...
        if fav_zoom_class.status == STARTED:
            # update params
            pass
        
        # *fav_a_friend* updates
        
        # if fav_a_friend is starting this frame...
        if fav_a_friend.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fav_a_friend.frameNStart = frameN  # exact frame index
            fav_a_friend.tStart = t  # local t and not account for scr refresh
            fav_a_friend.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fav_a_friend, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fav_a_friend.started')
            # update status
            fav_a_friend.status = STARTED
            fav_a_friend.setAutoDraw(True)
        
        # if fav_a_friend is active this frame...
        if fav_a_friend.status == STARTED:
            # update params
            pass
        
        # *fav_dissection_disaster* updates
        
        # if fav_dissection_disaster is starting this frame...
        if fav_dissection_disaster.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fav_dissection_disaster.frameNStart = frameN  # exact frame index
            fav_dissection_disaster.tStart = t  # local t and not account for scr refresh
            fav_dissection_disaster.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fav_dissection_disaster, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fav_dissection_disaster.started')
            # update status
            fav_dissection_disaster.status = STARTED
            fav_dissection_disaster.setAutoDraw(True)
        
        # if fav_dissection_disaster is active this frame...
        if fav_dissection_disaster.status == STARTED:
            # update params
            pass
        
        # *fav_party_prank* updates
        
        # if fav_party_prank is starting this frame...
        if fav_party_prank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fav_party_prank.frameNStart = frameN  # exact frame index
            fav_party_prank.tStart = t  # local t and not account for scr refresh
            fav_party_prank.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fav_party_prank, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fav_party_prank.started')
            # update status
            fav_party_prank.status = STARTED
            fav_party_prank.setAutoDraw(True)
        
        # if fav_party_prank is active this frame...
        if fav_party_prank.status == STARTED:
            # update params
            pass
        
        # *fav_a_tornado* updates
        
        # if fav_a_tornado is starting this frame...
        if fav_a_tornado.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fav_a_tornado.frameNStart = frameN  # exact frame index
            fav_a_tornado.tStart = t  # local t and not account for scr refresh
            fav_a_tornado.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fav_a_tornado, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fav_a_tornado.started')
            # update status
            fav_a_tornado.status = STARTED
            fav_a_tornado.setAutoDraw(True)
        
        # if fav_a_tornado is active this frame...
        if fav_a_tornado.status == STARTED:
            # update params
            pass
        
        # *fav_dance_contest_option* updates
        
        # if fav_dance_contest_option is starting this frame...
        if fav_dance_contest_option.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fav_dance_contest_option.frameNStart = frameN  # exact frame index
            fav_dance_contest_option.tStart = t  # local t and not account for scr refresh
            fav_dance_contest_option.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fav_dance_contest_option, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fav_dance_contest_option.started')
            # update status
            fav_dance_contest_option.status = STARTED
            fav_dance_contest_option.setAutoDraw(True)
        
        # if fav_dance_contest_option is active this frame...
        if fav_dance_contest_option.status == STARTED:
            # update params
            pass
        
        # *fav_zoom_class_option* updates
        
        # if fav_zoom_class_option is starting this frame...
        if fav_zoom_class_option.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fav_zoom_class_option.frameNStart = frameN  # exact frame index
            fav_zoom_class_option.tStart = t  # local t and not account for scr refresh
            fav_zoom_class_option.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fav_zoom_class_option, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fav_zoom_class_option.started')
            # update status
            fav_zoom_class_option.status = STARTED
            fav_zoom_class_option.setAutoDraw(True)
        
        # if fav_zoom_class_option is active this frame...
        if fav_zoom_class_option.status == STARTED:
            # update params
            pass
        
        # *fav_a_friend_option* updates
        
        # if fav_a_friend_option is starting this frame...
        if fav_a_friend_option.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fav_a_friend_option.frameNStart = frameN  # exact frame index
            fav_a_friend_option.tStart = t  # local t and not account for scr refresh
            fav_a_friend_option.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fav_a_friend_option, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fav_a_friend_option.started')
            # update status
            fav_a_friend_option.status = STARTED
            fav_a_friend_option.setAutoDraw(True)
        
        # if fav_a_friend_option is active this frame...
        if fav_a_friend_option.status == STARTED:
            # update params
            pass
        
        # *fav_dissection_disaster_option* updates
        
        # if fav_dissection_disaster_option is starting this frame...
        if fav_dissection_disaster_option.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fav_dissection_disaster_option.frameNStart = frameN  # exact frame index
            fav_dissection_disaster_option.tStart = t  # local t and not account for scr refresh
            fav_dissection_disaster_option.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fav_dissection_disaster_option, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fav_dissection_disaster_option.started')
            # update status
            fav_dissection_disaster_option.status = STARTED
            fav_dissection_disaster_option.setAutoDraw(True)
        
        # if fav_dissection_disaster_option is active this frame...
        if fav_dissection_disaster_option.status == STARTED:
            # update params
            pass
        
        # *fav_party_prank_option* updates
        
        # if fav_party_prank_option is starting this frame...
        if fav_party_prank_option.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fav_party_prank_option.frameNStart = frameN  # exact frame index
            fav_party_prank_option.tStart = t  # local t and not account for scr refresh
            fav_party_prank_option.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fav_party_prank_option, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fav_party_prank_option.started')
            # update status
            fav_party_prank_option.status = STARTED
            fav_party_prank_option.setAutoDraw(True)
        
        # if fav_party_prank_option is active this frame...
        if fav_party_prank_option.status == STARTED:
            # update params
            pass
        
        # *fav_a_tornado_option* updates
        
        # if fav_a_tornado_option is starting this frame...
        if fav_a_tornado_option.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fav_a_tornado_option.frameNStart = frameN  # exact frame index
            fav_a_tornado_option.tStart = t  # local t and not account for scr refresh
            fav_a_tornado_option.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fav_a_tornado_option, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fav_a_tornado_option.started')
            # update status
            fav_a_tornado_option.status = STARTED
            fav_a_tornado_option.setAutoDraw(True)
        
        # if fav_a_tornado_option is active this frame...
        if fav_a_tornado_option.status == STARTED:
            # update params
            pass
        # *fav_story_continue* updates
        
        # if fav_story_continue is active this frame...
        if fav_story_continue.status == STARTED:
            # update params
            pass
            # check whether fav_story_continue has been pressed
            if fav_story_continue.isClicked:
                if not fav_story_continue.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    fav_story_continue.timesOn.append(fav_story_continue.buttonClock.getTime())
                    fav_story_continue.timesOff.append(fav_story_continue.buttonClock.getTime())
                elif len(fav_story_continue.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    fav_story_continue.timesOff[-1] = fav_story_continue.buttonClock.getTime()
                if not fav_story_continue.wasClicked:
                    # end routine when fav_story_continue is clicked
                    continueRoutine = False
                if not fav_story_continue.wasClicked:
                    # run callback code when fav_story_continue is clicked
                    pass
        # take note of whether fav_story_continue was clicked, so that next frame we know if clicks are new
        fav_story_continue.wasClicked = fav_story_continue.isClicked and fav_story_continue.status == STARTED
        # *fav_mouse* updates
        
        # if fav_mouse is starting this frame...
        if fav_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fav_mouse.frameNStart = frameN  # exact frame index
            fav_mouse.tStart = t  # local t and not account for scr refresh
            fav_mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fav_mouse, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('fav_mouse.started', t)
            # update status
            fav_mouse.status = STARTED
            fav_mouse.mouseClock.reset()
            prevButtonState = fav_mouse.getPressed()  # if button is down already this ISN'T a new click
        if fav_mouse.status == STARTED:  # only update if started and not finished!
            buttons = fav_mouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames([fav_dance_contest_option, fav_zoom_class_option, fav_a_friend_option, fav_dissection_disaster_option, fav_party_prank_option, fav_a_tornado_option], namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(fav_mouse):
                            gotValidClick = True
                            fav_mouse.clicked_name.append(obj.name)
                    x, y = fav_mouse.getPos()
                    fav_mouse.x.append(x)
                    fav_mouse.y.append(y)
                    buttons = fav_mouse.getPressed()
                    fav_mouse.leftButton.append(buttons[0])
                    fav_mouse.midButton.append(buttons[1])
                    fav_mouse.rightButton.append(buttons[2])
                    fav_mouse.time.append(fav_mouse.mouseClock.getTime())
        # Run 'Each Frame' code from fav_story_code
        # Loop through each polygon and check for mouse clicks
        for choice in options:
            if fav_mouse.isPressedIn(choice):
                if most_favorite_story:
                    # Reset color of previously selected polygon
                    most_favorite_story.fillColor = 'white'
                # Set the new selection
                most_favorite_story = choice
                most_favorite_story.fillColor = 'red'
                fav_story_continue.setAutoDraw(True)
                break
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in favorite_storyComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "favorite_story" ---
    for thisComponent in favorite_storyComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('favorite_story.stopped', globalClock.getTime(format='float'))
    thisExp.addData('fav_story_continue.numClicks', fav_story_continue.numClicks)
    if fav_story_continue.numClicks:
       thisExp.addData('fav_story_continue.timesOn', fav_story_continue.timesOn)
       thisExp.addData('fav_story_continue.timesOff', fav_story_continue.timesOff)
    else:
       thisExp.addData('fav_story_continue.timesOn', "")
       thisExp.addData('fav_story_continue.timesOff', "")
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('fav_mouse.x', fav_mouse.x)
    thisExp.addData('fav_mouse.y', fav_mouse.y)
    thisExp.addData('fav_mouse.leftButton', fav_mouse.leftButton)
    thisExp.addData('fav_mouse.midButton', fav_mouse.midButton)
    thisExp.addData('fav_mouse.rightButton', fav_mouse.rightButton)
    thisExp.addData('fav_mouse.time', fav_mouse.time)
    thisExp.addData('fav_mouse.clicked_name', fav_mouse.clicked_name)
    # Run 'End Routine' code from fav_story_code
    import time
    from pylsl import local_clock
    
    win.mouseVisible = False
    marker_time = time.time()
    
    if most_favorite_story:
        thisExp.addData('most_favorite_story', most_favorite_story.name)
    else:
        thisExp.addData('most_favorite_story', 'None')
    
    
    if most_favorite_story.name == 'fav_a_friend_option':
        outlet.push_sample([22])
        outlet.push_sample([marker_time])
    elif most_favorite_story.name == 'fav_dissection_disaster_option':
        outlet.push_sample([32])
        outlet.push_sample([marker_time])
    elif most_favorite_story.name == 'fav_dance_contest_option':
        outlet.push_sample([42])
        outlet.push_sample([marker_time])
    elif most_favorite_story.name == 'fav_zoom_class_option':
        outlet.push_sample([52])
        outlet.push_sample([marker_time])
    elif most_favorite_story.name == 'fav_a_tornado_option':
        outlet.push_sample([62])
        outlet.push_sample([marker_time])
    elif most_favorite_story.name == 'fav_party_prank_option':
        outlet.push_sample([72])
        outlet.push_sample([marker_time])
    
    outlet.push_sample([303])
    outlet.push_sample([marker_time])
    
    thisExp.nextEntry()
    # the Routine "favorite_story" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "least_favorite_story" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('least_favorite_story.started', globalClock.getTime(format='float'))
    # Run 'Begin Routine' code from align_text_2
    least_fav_story_names.alignText = 'left'
    # reset least_fav_story_continue to account for continued clicks & clear times on/off
    least_fav_story_continue.reset()
    # setup some python lists for storing info about the least_fav_mouse
    least_fav_mouse.x = []
    least_fav_mouse.y = []
    least_fav_mouse.leftButton = []
    least_fav_mouse.midButton = []
    least_fav_mouse.rightButton = []
    least_fav_mouse.time = []
    least_fav_mouse.clicked_name = []
    gotValidClick = False  # until a click is received
    # Run 'Begin Routine' code from least_fav_story_code
    import time
    from pylsl import local_clock
    
    win.mouseVisible = True
    marker_time = time.time()
    
    least_favorite_story = None
    options = [least_fav_dance_contest_option, least_fav_zoom_class_option, least_fav_a_friend_option, least_fav_dissection_disaster_option, least_fav_party_prank_option, least_fav_a_tornado_option]
    
    outlet.push_sample([304])
    outlet.push_sample([marker_time])
    
    
    # keep track of which components have finished
    least_favorite_storyComponents = [least_fav, least_fav_story_names, least_fav_dance_contest, least_fav_zoom_class, least_fav_a_friend, least_fav_dissection_disaster, least_fav_party_prank, least_fav_a_tornado, least_fav_dance_contest_option, least_fav_zoom_class_option, least_fav_a_friend_option, least_fav_dissection_disaster_option, least_fav_party_prank_option, least_fav_a_tornado_option, least_fav_story_continue, least_fav_mouse]
    for thisComponent in least_favorite_storyComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "least_favorite_story" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *least_fav* updates
        
        # if least_fav is starting this frame...
        if least_fav.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            least_fav.frameNStart = frameN  # exact frame index
            least_fav.tStart = t  # local t and not account for scr refresh
            least_fav.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(least_fav, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'least_fav.started')
            # update status
            least_fav.status = STARTED
            least_fav.setAutoDraw(True)
        
        # if least_fav is active this frame...
        if least_fav.status == STARTED:
            # update params
            pass
        
        # *least_fav_story_names* updates
        
        # if least_fav_story_names is starting this frame...
        if least_fav_story_names.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            least_fav_story_names.frameNStart = frameN  # exact frame index
            least_fav_story_names.tStart = t  # local t and not account for scr refresh
            least_fav_story_names.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(least_fav_story_names, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'least_fav_story_names.started')
            # update status
            least_fav_story_names.status = STARTED
            least_fav_story_names.setAutoDraw(True)
        
        # if least_fav_story_names is active this frame...
        if least_fav_story_names.status == STARTED:
            # update params
            pass
        
        # *least_fav_dance_contest* updates
        
        # if least_fav_dance_contest is starting this frame...
        if least_fav_dance_contest.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            least_fav_dance_contest.frameNStart = frameN  # exact frame index
            least_fav_dance_contest.tStart = t  # local t and not account for scr refresh
            least_fav_dance_contest.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(least_fav_dance_contest, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'least_fav_dance_contest.started')
            # update status
            least_fav_dance_contest.status = STARTED
            least_fav_dance_contest.setAutoDraw(True)
        
        # if least_fav_dance_contest is active this frame...
        if least_fav_dance_contest.status == STARTED:
            # update params
            pass
        
        # *least_fav_zoom_class* updates
        
        # if least_fav_zoom_class is starting this frame...
        if least_fav_zoom_class.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            least_fav_zoom_class.frameNStart = frameN  # exact frame index
            least_fav_zoom_class.tStart = t  # local t and not account for scr refresh
            least_fav_zoom_class.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(least_fav_zoom_class, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'least_fav_zoom_class.started')
            # update status
            least_fav_zoom_class.status = STARTED
            least_fav_zoom_class.setAutoDraw(True)
        
        # if least_fav_zoom_class is active this frame...
        if least_fav_zoom_class.status == STARTED:
            # update params
            pass
        
        # *least_fav_a_friend* updates
        
        # if least_fav_a_friend is starting this frame...
        if least_fav_a_friend.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            least_fav_a_friend.frameNStart = frameN  # exact frame index
            least_fav_a_friend.tStart = t  # local t and not account for scr refresh
            least_fav_a_friend.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(least_fav_a_friend, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'least_fav_a_friend.started')
            # update status
            least_fav_a_friend.status = STARTED
            least_fav_a_friend.setAutoDraw(True)
        
        # if least_fav_a_friend is active this frame...
        if least_fav_a_friend.status == STARTED:
            # update params
            pass
        
        # *least_fav_dissection_disaster* updates
        
        # if least_fav_dissection_disaster is starting this frame...
        if least_fav_dissection_disaster.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            least_fav_dissection_disaster.frameNStart = frameN  # exact frame index
            least_fav_dissection_disaster.tStart = t  # local t and not account for scr refresh
            least_fav_dissection_disaster.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(least_fav_dissection_disaster, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'least_fav_dissection_disaster.started')
            # update status
            least_fav_dissection_disaster.status = STARTED
            least_fav_dissection_disaster.setAutoDraw(True)
        
        # if least_fav_dissection_disaster is active this frame...
        if least_fav_dissection_disaster.status == STARTED:
            # update params
            pass
        
        # *least_fav_party_prank* updates
        
        # if least_fav_party_prank is starting this frame...
        if least_fav_party_prank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            least_fav_party_prank.frameNStart = frameN  # exact frame index
            least_fav_party_prank.tStart = t  # local t and not account for scr refresh
            least_fav_party_prank.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(least_fav_party_prank, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'least_fav_party_prank.started')
            # update status
            least_fav_party_prank.status = STARTED
            least_fav_party_prank.setAutoDraw(True)
        
        # if least_fav_party_prank is active this frame...
        if least_fav_party_prank.status == STARTED:
            # update params
            pass
        
        # *least_fav_a_tornado* updates
        
        # if least_fav_a_tornado is starting this frame...
        if least_fav_a_tornado.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            least_fav_a_tornado.frameNStart = frameN  # exact frame index
            least_fav_a_tornado.tStart = t  # local t and not account for scr refresh
            least_fav_a_tornado.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(least_fav_a_tornado, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'least_fav_a_tornado.started')
            # update status
            least_fav_a_tornado.status = STARTED
            least_fav_a_tornado.setAutoDraw(True)
        
        # if least_fav_a_tornado is active this frame...
        if least_fav_a_tornado.status == STARTED:
            # update params
            pass
        
        # *least_fav_dance_contest_option* updates
        
        # if least_fav_dance_contest_option is starting this frame...
        if least_fav_dance_contest_option.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            least_fav_dance_contest_option.frameNStart = frameN  # exact frame index
            least_fav_dance_contest_option.tStart = t  # local t and not account for scr refresh
            least_fav_dance_contest_option.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(least_fav_dance_contest_option, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'least_fav_dance_contest_option.started')
            # update status
            least_fav_dance_contest_option.status = STARTED
            least_fav_dance_contest_option.setAutoDraw(True)
        
        # if least_fav_dance_contest_option is active this frame...
        if least_fav_dance_contest_option.status == STARTED:
            # update params
            pass
        
        # *least_fav_zoom_class_option* updates
        
        # if least_fav_zoom_class_option is starting this frame...
        if least_fav_zoom_class_option.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            least_fav_zoom_class_option.frameNStart = frameN  # exact frame index
            least_fav_zoom_class_option.tStart = t  # local t and not account for scr refresh
            least_fav_zoom_class_option.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(least_fav_zoom_class_option, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'least_fav_zoom_class_option.started')
            # update status
            least_fav_zoom_class_option.status = STARTED
            least_fav_zoom_class_option.setAutoDraw(True)
        
        # if least_fav_zoom_class_option is active this frame...
        if least_fav_zoom_class_option.status == STARTED:
            # update params
            pass
        
        # *least_fav_a_friend_option* updates
        
        # if least_fav_a_friend_option is starting this frame...
        if least_fav_a_friend_option.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            least_fav_a_friend_option.frameNStart = frameN  # exact frame index
            least_fav_a_friend_option.tStart = t  # local t and not account for scr refresh
            least_fav_a_friend_option.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(least_fav_a_friend_option, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'least_fav_a_friend_option.started')
            # update status
            least_fav_a_friend_option.status = STARTED
            least_fav_a_friend_option.setAutoDraw(True)
        
        # if least_fav_a_friend_option is active this frame...
        if least_fav_a_friend_option.status == STARTED:
            # update params
            pass
        
        # *least_fav_dissection_disaster_option* updates
        
        # if least_fav_dissection_disaster_option is starting this frame...
        if least_fav_dissection_disaster_option.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            least_fav_dissection_disaster_option.frameNStart = frameN  # exact frame index
            least_fav_dissection_disaster_option.tStart = t  # local t and not account for scr refresh
            least_fav_dissection_disaster_option.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(least_fav_dissection_disaster_option, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'least_fav_dissection_disaster_option.started')
            # update status
            least_fav_dissection_disaster_option.status = STARTED
            least_fav_dissection_disaster_option.setAutoDraw(True)
        
        # if least_fav_dissection_disaster_option is active this frame...
        if least_fav_dissection_disaster_option.status == STARTED:
            # update params
            pass
        
        # *least_fav_party_prank_option* updates
        
        # if least_fav_party_prank_option is starting this frame...
        if least_fav_party_prank_option.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            least_fav_party_prank_option.frameNStart = frameN  # exact frame index
            least_fav_party_prank_option.tStart = t  # local t and not account for scr refresh
            least_fav_party_prank_option.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(least_fav_party_prank_option, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'least_fav_party_prank_option.started')
            # update status
            least_fav_party_prank_option.status = STARTED
            least_fav_party_prank_option.setAutoDraw(True)
        
        # if least_fav_party_prank_option is active this frame...
        if least_fav_party_prank_option.status == STARTED:
            # update params
            pass
        
        # *least_fav_a_tornado_option* updates
        
        # if least_fav_a_tornado_option is starting this frame...
        if least_fav_a_tornado_option.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            least_fav_a_tornado_option.frameNStart = frameN  # exact frame index
            least_fav_a_tornado_option.tStart = t  # local t and not account for scr refresh
            least_fav_a_tornado_option.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(least_fav_a_tornado_option, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'least_fav_a_tornado_option.started')
            # update status
            least_fav_a_tornado_option.status = STARTED
            least_fav_a_tornado_option.setAutoDraw(True)
        
        # if least_fav_a_tornado_option is active this frame...
        if least_fav_a_tornado_option.status == STARTED:
            # update params
            pass
        # *least_fav_story_continue* updates
        
        # if least_fav_story_continue is active this frame...
        if least_fav_story_continue.status == STARTED:
            # update params
            pass
            # check whether least_fav_story_continue has been pressed
            if least_fav_story_continue.isClicked:
                if not least_fav_story_continue.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    least_fav_story_continue.timesOn.append(least_fav_story_continue.buttonClock.getTime())
                    least_fav_story_continue.timesOff.append(least_fav_story_continue.buttonClock.getTime())
                elif len(least_fav_story_continue.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    least_fav_story_continue.timesOff[-1] = least_fav_story_continue.buttonClock.getTime()
                if not least_fav_story_continue.wasClicked:
                    # end routine when least_fav_story_continue is clicked
                    continueRoutine = False
                if not least_fav_story_continue.wasClicked:
                    # run callback code when least_fav_story_continue is clicked
                    pass
        # take note of whether least_fav_story_continue was clicked, so that next frame we know if clicks are new
        least_fav_story_continue.wasClicked = least_fav_story_continue.isClicked and least_fav_story_continue.status == STARTED
        # *least_fav_mouse* updates
        
        # if least_fav_mouse is starting this frame...
        if least_fav_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            least_fav_mouse.frameNStart = frameN  # exact frame index
            least_fav_mouse.tStart = t  # local t and not account for scr refresh
            least_fav_mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(least_fav_mouse, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('least_fav_mouse.started', t)
            # update status
            least_fav_mouse.status = STARTED
            least_fav_mouse.mouseClock.reset()
            prevButtonState = least_fav_mouse.getPressed()  # if button is down already this ISN'T a new click
        if least_fav_mouse.status == STARTED:  # only update if started and not finished!
            buttons = least_fav_mouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames([least_fav_dance_contest_option, least_fav_zoom_class_option, least_fav_a_friend_option, least_fav_dissection_disaster_option, least_fav_party_prank_option, least_fav_a_tornado_option], namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(least_fav_mouse):
                            gotValidClick = True
                            least_fav_mouse.clicked_name.append(obj.name)
                    x, y = least_fav_mouse.getPos()
                    least_fav_mouse.x.append(x)
                    least_fav_mouse.y.append(y)
                    buttons = least_fav_mouse.getPressed()
                    least_fav_mouse.leftButton.append(buttons[0])
                    least_fav_mouse.midButton.append(buttons[1])
                    least_fav_mouse.rightButton.append(buttons[2])
                    least_fav_mouse.time.append(least_fav_mouse.mouseClock.getTime())
        # Run 'Each Frame' code from least_fav_story_code
        # Loop through each polygon and check for mouse clicks
        for choice in options:
            if least_fav_mouse.isPressedIn(choice):
                if least_favorite_story:
                    # Reset color of previously selected polygon
                    least_favorite_story.fillColor = 'white'
                # Set the new selection
                least_favorite_story = choice
                least_favorite_story.fillColor = 'red'
                least_fav_story_continue.setAutoDraw(True)
                break
        
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in least_favorite_storyComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "least_favorite_story" ---
    for thisComponent in least_favorite_storyComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('least_favorite_story.stopped', globalClock.getTime(format='float'))
    thisExp.addData('least_fav_story_continue.numClicks', least_fav_story_continue.numClicks)
    if least_fav_story_continue.numClicks:
       thisExp.addData('least_fav_story_continue.timesOn', least_fav_story_continue.timesOn)
       thisExp.addData('least_fav_story_continue.timesOff', least_fav_story_continue.timesOff)
    else:
       thisExp.addData('least_fav_story_continue.timesOn', "")
       thisExp.addData('least_fav_story_continue.timesOff', "")
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('least_fav_mouse.x', least_fav_mouse.x)
    thisExp.addData('least_fav_mouse.y', least_fav_mouse.y)
    thisExp.addData('least_fav_mouse.leftButton', least_fav_mouse.leftButton)
    thisExp.addData('least_fav_mouse.midButton', least_fav_mouse.midButton)
    thisExp.addData('least_fav_mouse.rightButton', least_fav_mouse.rightButton)
    thisExp.addData('least_fav_mouse.time', least_fav_mouse.time)
    thisExp.addData('least_fav_mouse.clicked_name', least_fav_mouse.clicked_name)
    # Run 'End Routine' code from least_fav_story_code
    import time
    from pylsl import local_clock
    
    win.mouseVisible = False
    marker_time = time.time()
    
    if least_favorite_story:
        thisExp.addData('least_favorite_story', least_favorite_story.name)
    else:
        thisExp.addData('least_favorite_story', 'None')
    
    if least_favorite_story.name == 'least_fav_a_friend_option':
        outlet.push_sample([22])
        outlet.push_sample([marker_time])
    elif least_favorite_story.name == 'least_fav_dissection_disaster_option':
        outlet.push_sample([32])
        outlet.push_sample([marker_time])
    elif least_favorite_story.name == 'least_fav_dance_contest_option':
        outlet.push_sample([42])
        outlet.push_sample([marker_time])
    elif least_favorite_story.name == 'least_fav_zoom_class_option':
        outlet.push_sample([52])
        outlet.push_sample([int(marker_time)])
    elif least_favorite_story.name == 'least_fav_a_tornado_option':
        outlet.push_sample([62])
        outlet.push_sample([marker_time])
    elif least_favorite_story.name == 'least_fav_party_prank_option':
        outlet.push_sample([72])
        outlet.push_sample([marker_time])
    
    outlet.push_sample([305])
    outlet.push_sample([marker_time])
    outlet.push_sample([501])
    
    thisExp.nextEntry()
    # the Routine "least_favorite_story" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "check_impedance" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('check_impedance.started', globalClock.getTime(format='float'))
    impedance_end.keys = []
    impedance_end.rt = []
    _impedance_end_allKeys = []
    # Run 'Begin Routine' code from check_impedance_code
    import time
    from pylsl import local_clock
    
    win.mouseVisible = False
    marker_time = time.time()
    
    outlet.push_sample([400])
    outlet.push_sample([marker_time])
    # keep track of which components have finished
    check_impedanceComponents = [impedance_pause, impedance_end]
    for thisComponent in check_impedanceComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "check_impedance" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *impedance_pause* updates
        
        # if impedance_pause is starting this frame...
        if impedance_pause.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            impedance_pause.frameNStart = frameN  # exact frame index
            impedance_pause.tStart = t  # local t and not account for scr refresh
            impedance_pause.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(impedance_pause, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'impedance_pause.started')
            # update status
            impedance_pause.status = STARTED
            impedance_pause.setAutoDraw(True)
        
        # if impedance_pause is active this frame...
        if impedance_pause.status == STARTED:
            # update params
            pass
        
        # *impedance_end* updates
        waitOnFlip = False
        
        # if impedance_end is starting this frame...
        if impedance_end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            impedance_end.frameNStart = frameN  # exact frame index
            impedance_end.tStart = t  # local t and not account for scr refresh
            impedance_end.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(impedance_end, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'impedance_end.started')
            # update status
            impedance_end.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(impedance_end.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(impedance_end.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if impedance_end.status == STARTED and not waitOnFlip:
            theseKeys = impedance_end.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _impedance_end_allKeys.extend(theseKeys)
            if len(_impedance_end_allKeys):
                impedance_end.keys = _impedance_end_allKeys[-1].name  # just the last key pressed
                impedance_end.rt = _impedance_end_allKeys[-1].rt
                impedance_end.duration = _impedance_end_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in check_impedanceComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "check_impedance" ---
    for thisComponent in check_impedanceComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('check_impedance.stopped', globalClock.getTime(format='float'))
    # check responses
    if impedance_end.keys in ['', [], None]:  # No response was made
        impedance_end.keys = None
    thisExp.addData('impedance_end.keys',impedance_end.keys)
    if impedance_end.keys != None:  # we had a response
        thisExp.addData('impedance_end.rt', impedance_end.rt)
        thisExp.addData('impedance_end.duration', impedance_end.duration)
    # Run 'End Routine' code from check_impedance_code
    import time
    from pylsl import local_clock
    
    win.mouseVisible = True
    marker_time = time.time()
    
    outlet.push_sample([401])
    outlet.push_sample([marker_time])
    thisExp.nextEntry()
    # the Routine "check_impedance" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "social_welcome" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('social_welcome.started', globalClock.getTime(format='float'))
    # reset social_start_button to account for continued clicks & clear times on/off
    social_start_button.reset()
    # setup some python lists for storing info about the mouse
    mouse.x = []
    mouse.y = []
    mouse.leftButton = []
    mouse.midButton = []
    mouse.rightButton = []
    mouse.time = []
    mouse.clicked_name = []
    gotValidClick = False  # until a click is received
    # Run 'Begin Routine' code from code
    win.mouseVisible = True
    # keep track of which components have finished
    social_welcomeComponents = [text_3, social_start_button, mouse]
    for thisComponent in social_welcomeComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "social_welcome" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_3* updates
        
        # if text_3 is starting this frame...
        if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_3.frameNStart = frameN  # exact frame index
            text_3.tStart = t  # local t and not account for scr refresh
            text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_3.started')
            # update status
            text_3.status = STARTED
            text_3.setAutoDraw(True)
        
        # if text_3 is active this frame...
        if text_3.status == STARTED:
            # update params
            pass
        # *social_start_button* updates
        
        # if social_start_button is starting this frame...
        if social_start_button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            social_start_button.frameNStart = frameN  # exact frame index
            social_start_button.tStart = t  # local t and not account for scr refresh
            social_start_button.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(social_start_button, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'social_start_button.started')
            # update status
            social_start_button.status = STARTED
            social_start_button.setAutoDraw(True)
        
        # if social_start_button is active this frame...
        if social_start_button.status == STARTED:
            # update params
            pass
            # check whether social_start_button has been pressed
            if social_start_button.isClicked:
                if not social_start_button.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    social_start_button.timesOn.append(social_start_button.buttonClock.getTime())
                    social_start_button.timesOff.append(social_start_button.buttonClock.getTime())
                elif len(social_start_button.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    social_start_button.timesOff[-1] = social_start_button.buttonClock.getTime()
                if not social_start_button.wasClicked:
                    # end routine when social_start_button is clicked
                    continueRoutine = False
                if not social_start_button.wasClicked:
                    # run callback code when social_start_button is clicked
                    pass
        # take note of whether social_start_button was clicked, so that next frame we know if clicks are new
        social_start_button.wasClicked = social_start_button.isClicked and social_start_button.status == STARTED
        # *mouse* updates
        
        # if mouse is starting this frame...
        if mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse.frameNStart = frameN  # exact frame index
            mouse.tStart = t  # local t and not account for scr refresh
            mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('mouse.started', t)
            # update status
            mouse.status = STARTED
            mouse.mouseClock.reset()
            prevButtonState = [0, 0, 0]  # if now button is down we will treat as 'new' click
        
        # if mouse is stopping this frame...
        if mouse.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > mouse.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                mouse.tStop = t  # not accounting for scr refresh
                mouse.tStopRefresh = tThisFlipGlobal  # on global time
                mouse.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.addData('mouse.stopped', t)
                # update status
                mouse.status = FINISHED
        if mouse.status == STARTED:  # only update if started and not finished!
            buttons = mouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(social_start_button, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse):
                            gotValidClick = True
                            mouse.clicked_name.append(obj.name)
                    x, y = mouse.getPos()
                    mouse.x.append(x)
                    mouse.y.append(y)
                    buttons = mouse.getPressed()
                    mouse.leftButton.append(buttons[0])
                    mouse.midButton.append(buttons[1])
                    mouse.rightButton.append(buttons[2])
                    mouse.time.append(mouse.mouseClock.getTime())
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in social_welcomeComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "social_welcome" ---
    for thisComponent in social_welcomeComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('social_welcome.stopped', globalClock.getTime(format='float'))
    thisExp.addData('social_start_button.numClicks', social_start_button.numClicks)
    if social_start_button.numClicks:
       thisExp.addData('social_start_button.timesOn', social_start_button.timesOn)
       thisExp.addData('social_start_button.timesOff', social_start_button.timesOff)
    else:
       thisExp.addData('social_start_button.timesOn', "")
       thisExp.addData('social_start_button.timesOff', "")
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse.x', mouse.x)
    thisExp.addData('mouse.y', mouse.y)
    thisExp.addData('mouse.leftButton', mouse.leftButton)
    thisExp.addData('mouse.midButton', mouse.midButton)
    thisExp.addData('mouse.rightButton', mouse.rightButton)
    thisExp.addData('mouse.time', mouse.time)
    thisExp.addData('mouse.clicked_name', mouse.clicked_name)
    thisExp.nextEntry()
    # the Routine "social_welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "social_starts" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('social_starts.started', globalClock.getTime(format='float'))
    social_end_sound.setSound('audio_files/normalized/new_beep_sound.wav', hamming=True)
    social_end_sound.setVolume(0.8, log=False)
    social_end_sound.seek(0)
    # Run 'Begin Routine' code from social_lsl
    import time
    from pylsl import local_clock
    
    marker_time = time.time()
    
    outlet.push_sample([80])
    outlet.push_sample([marker_time])
    # keep track of which components have finished
    social_startsComponents = [countdown, social_end_sound]
    for thisComponent in social_startsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "social_starts" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *countdown* updates
        
        # if countdown is starting this frame...
        if countdown.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            countdown.frameNStart = frameN  # exact frame index
            countdown.tStart = t  # local t and not account for scr refresh
            countdown.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(countdown, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'countdown.started')
            # update status
            countdown.status = STARTED
            countdown.setAutoDraw(True)
        
        # if countdown is active this frame...
        if countdown.status == STARTED:
            # update params
            pass
        
        # if countdown is stopping this frame...
        if countdown.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > countdown.tStartRefresh + 300-frameTolerance:
                # keep track of stop time/frame for later
                countdown.tStop = t  # not accounting for scr refresh
                countdown.tStopRefresh = tThisFlipGlobal  # on global time
                countdown.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'countdown.stopped')
                # update status
                countdown.status = FINISHED
                countdown.setAutoDraw(False)
        
        # if social_end_sound is starting this frame...
        if social_end_sound.status == NOT_STARTED and t >= 300-frameTolerance:
            # keep track of start time/frame for later
            social_end_sound.frameNStart = frameN  # exact frame index
            social_end_sound.tStart = t  # local t and not account for scr refresh
            social_end_sound.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('social_end_sound.started', t)
            # update status
            social_end_sound.status = STARTED
            social_end_sound.play()  # start the sound (it finishes automatically)
        # update social_end_sound status according to whether it's playing
        if social_end_sound.isPlaying:
            social_end_sound.status = STARTED
        elif social_end_sound.isFinished:
            social_end_sound.status = FINISHED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in social_startsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "social_starts" ---
    for thisComponent in social_startsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('social_starts.stopped', globalClock.getTime(format='float'))
    social_end_sound.pause()  # ensure sound has stopped at end of Routine
    # Run 'End Routine' code from social_lsl
    import time
    from pylsl import local_clock
    
    marker_time = time.time()
    
    outlet.push_sample([81])
    outlet.push_sample([marker_time])
    thisExp.nextEntry()
    # the Routine "social_starts" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "end_experiment" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('end_experiment.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    end_experimentComponents = [thank_you]
    for thisComponent in end_experimentComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end_experiment" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *thank_you* updates
        
        # if thank_you is starting this frame...
        if thank_you.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            thank_you.frameNStart = frameN  # exact frame index
            thank_you.tStart = t  # local t and not account for scr refresh
            thank_you.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(thank_you, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'thank_you.started')
            # update status
            thank_you.status = STARTED
            thank_you.setAutoDraw(True)
        
        # if thank_you is active this frame...
        if thank_you.status == STARTED:
            # update params
            pass
        
        # if thank_you is stopping this frame...
        if thank_you.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > thank_you.tStartRefresh + 5.0-frameTolerance:
                # keep track of stop time/frame for later
                thank_you.tStop = t  # not accounting for scr refresh
                thank_you.tStopRefresh = tThisFlipGlobal  # on global time
                thank_you.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'thank_you.stopped')
                # update status
                thank_you.status = FINISHED
                thank_you.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end_experimentComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end_experiment" ---
    for thisComponent in end_experimentComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('end_experiment.stopped', globalClock.getTime(format='float'))
    # Run 'End Routine' code from end_lsl
    import time
    from pylsl import local_clock
    
    marker_time = time.time()
    
    outlet.push_sample([201])
    outlet.push_sample([marker_time])
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    thisExp.nextEntry()
    # Run 'End Experiment' code from least_fav_story_code
    import time
    from pylsl import local_clock
    
    win.mouseVisible = False
    marker_time = time.time()
    
    outlet.push_sample([4])
    outlet.push_sample([marker_time])
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)

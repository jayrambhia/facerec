import cv2
import warnings
import os
import collections
import numpy as np

CASCADE = "face.xml"
SAMPLES_DIREC = "samples"
LAUNCH_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
CASCADE_PATH = os.path.join(LAUNCH_PATH, SAMPLES_DIREC, CASCADE)
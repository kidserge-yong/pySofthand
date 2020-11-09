import glob
import os
import sys
import threading

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

try:
    import pygame
    import pygame_menu
    import pygame_gui
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    sys.path.append('../module/')
    from qbrobot import *
    from smk import *
    from synergy import *
    from menu import *

except ImportError:
    raise RuntimeError('cannot import module, make sure sys.path is correct')

# ==============================================================================
# -- record() ------------------------------------------------------------------
# ==============================================================================

def record(device:smk_arrayemg, record_time:float, set_value:list):
    INTERUPT_FLAG = False
    t_end = time.time() + record_time
    data = []
    try:
        print("Use Ctrl + C to emergency stop recording.")
        while time.time() < t_end:
            if INTERUPT_FLAG:
                break
            if device.new_data:
                data.append(device.emg() + set_value)
            
        print("Finish rest state.")
    except KeyboardInterrupt:
        INTERUPT_FLAG = True
    return data, INTERUPT_FLAG


# ==============================================================================
# -- main_loop() ---------------------------------------------------------------
# ==============================================================================

def main_loop(args=0):
    """
    Softhand handle
    """
    softhand = robot(args.robot_port)
    softhand.add_device(1, "Hand Grip/Open", "softhand")
    softhand.add_device(2, "Wrist Flex/Exten", "qbmove")
    softhand.add_device(3, "Wrist Pron/Supi", "qbmove")
    softhand.start()
    softhand.start_lsl()
    """
    32ch SMK handle
    """
    smk = smk_arrayemg(args.sensor_port)
    smk.calibrate()
    smk.start()
    smk.start_lsl()
    """
    Synergy handle
    """
    synergy = synergy()

    """
    Collect Training data
    """
    ExperimentSequence = {
        0:("Rest", 5, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        1:("Grip 100%", 5, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        2:("Open 100%", 5, [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        3:("Wrist Flex 100%", 5, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        4:("Wrist Exten 100%", 5, [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]),
        5:("Pronation 100%", 5, [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        6:("Supination 100%", 5, [0.0, 0.0, 0.0, 0.0, -1.0, 0.0]),
    }
    sectional_data = [0]*len(ExperimentSequence)
    train_data = []
    for state, i in zip(ExperimentSequence, range(len(ExperimentSequence))):
        print(ExperimentSequence[state][0])
        input("Please press any key to start record state.")
        new_data, flag = record(smk, ExperimentSequence[state][1], ExperimentSequence[state][2])
        train_data = train_data + new_data
        sectional_data[i] = new_data
        if flag == True:
            break

    assert len(train_data) != len(ExperimentSequence), "Problem with data collection"

    """
    Stop process
    """
    softhand.stop_lsl()
    softhand.stop()
    smk.stop_lsl()
    smk.stop()

    


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def main():
    argparser = argparse.ArgumentParser(
        description='Synergy with smk array emg and softhand demonstration')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '-rp', '--robot_port',
        metavar='P',
        help='Serial port connect to softhand robot')
    argparser.add_argument(
        '-sp', '--sensor_port',
        metavar='P',
        help='Serial port connect to emg sensor')
    argparser.add_argument(
        '--title',
        metavar='device_name',
        default='Softhand Robot',
        help='The title to support user')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    args = connect_menu(args)
    logging.info('listening to device %s on port %s', args.title, args.port)
    args.robot_port = args.port
    args.title = "SMK 32ch Array EMG"

    args = connect_menu(args)
    logging.info('listening to device %s on port %s', args.title, args.port)
    args.sensor_port = args.port

    print(__doc__)

    try:
        main_loop(args)
        #pass

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
    #main_loop()

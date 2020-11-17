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

import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

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

tp = lambda n: np.transpose(n)

# ==============================================================================
# -- ETC function() ------------------------------------------------------------
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
            if device.is_new_data:
                data.append(device.emg() + set_value)
            
        print("Finish rest state.")
    except KeyboardInterrupt:
        INTERUPT_FLAG = True
    return data, INTERUPT_FLAG

def plot(graph):
    plt.plot(graph)
    plt.show() 

def multiplot(graph, channel_dir = 1):
    if channel_dir > 0:
        graph = tp(graph)
    fig, axs = plt.subplots(len(graph), 1)
    for i, item in enumerate(graph):
        axs[i].plot(item)
    plt.show()


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
    syn = synergy()
    regressor = LinearRegression()
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
    sectional_data = []
    train_data = []
    for state, i in zip(ExperimentSequence, range(len(ExperimentSequence))):
        print(ExperimentSequence[state][0])
        input("Please press any key to start record state.")
        new_data, flag = record(smk, ExperimentSequence[state][1], ExperimentSequence[state][2])
        train_data = train_data + new_data
        sectional_data.append(new_data)
        if flag == True:
            break

    assert len(sectional_data) == len(ExperimentSequence), "Problem with data collection"

    """
    Training synergy model
    """
    EMGdata = []
    Motiondata = []
    for section in sectional_data:
        EMGdata.append(tp([item[:-6] for item in section]))
        Motiondata.append(tp([item[-6:] for item in section]))

    print(len(EMGdata), len(EMGdata[0]))
    print(len(Motiondata), len(Motiondata[0]))


    gripdata_X = np.concatenate((EMGdata[0], EMGdata[1], EMGdata[2]), axis=1)
    gripdata_Y = np.concatenate((Motiondata[0], Motiondata[1], Motiondata[2]), axis=1)

    wristdata_X = np.concatenate((EMGdata[0], EMGdata[3], EMGdata[4]), axis=1)
    wristdata_Y = np.concatenate((Motiondata[0], Motiondata[3], Motiondata[4]), axis=1)

    prosudata_X = np.concatenate((EMGdata[0], EMGdata[5], EMGdata[6]), axis=1)
    prosudata_Y = np.concatenate((Motiondata[0], Motiondata[5], Motiondata[6]), axis=1)

    sum_x = [gripdata_X, wristdata_X, prosudata_X]
    sum_y = [gripdata_Y, wristdata_Y, prosudata_Y]

    syn.fit(sum_x, sum_y)

    xsynergy = tp(syn.transform(tp([item[:-6] for item in train_data])))
    nY = [item[-6:] for item in train_data]
    

    multiplot(xsynergy)
    multiplot(nY)
    

    #print(len(xsynergy), len(xsynergy[0]))
    #print(len(nY), len(nY[0]))

    regressor.fit(xsynergy, nY)

    angle = regressor.predict(xsynergy)
    multiplot(angle)

    """
    Testing model
    """
    run_data = []
    print("Start main testing loop")
    try:
        while(True):
            if smk.is_new_data:
                xemg = [smk.emg()]
                xsyn = tp(syn.transform(tp(xemg)))
                #xsyn = syn.transform(tp(xemg))
                angle = regressor.predict(xsyn)

                grip_pos = angle[0][0]*19000 + 0
                #grip_pos = position[0][0]*20000 +3000
                if grip_pos < 0:
                    grip_pos = 0
                elif grip_pos > 19000:
                    grip_pos = 19000

                wrist_pos = (angle[0][2]*10000)-5000
                #wrist_pos = (position[0][2]*10000)-5000
                if wrist_pos < -5000 :
                    wrist_pos = -5000
                elif wrist_pos > 5000:
                    wrist_pos = 5000

                prosu_pos = (angle[0][4]*10000)-5000
                #wrist_pos = (position[0][2]*10000)-5000
                if prosu_pos < -5000 :
                    prosu_pos = -5000
                elif prosu_pos > 5000:
                    prosu_pos = 5000

                softhand.movedevice(0, int(grip_pos))
                softhand.movedevice(1, int(wrist_pos))
                softhand.movedevice(2, int(prosu_pos))
                run_data.append([xemg[0], angle[0]])
    except KeyboardInterrupt:
        print("out of loop")


    """
    Saving
    """
    try:
        smk.save2File(train_data, filename = "../data/train_data_"+time.strftime("%m_%d_%Y_%H_%M_%S")+".csv")
        smk.save2File(run_data, filename = "../data/run_data_"+time.strftime("%m_%d_%Y_%H_%M_%S")+".csv")
    except:
        pass

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

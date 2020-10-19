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
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import serial
except ImportError:
    raise RuntimeError('cannot import serial, make sure pyserial package is installed')

try:
    sys.path.append('../module/')
    from menu import *
except ImportError:
    raise RuntimeError('cannot import module, make sure sys.path is correct')

# ==============================================================================
# -- main_loop() ---------------------------------------------------------------
# ==============================================================================

def main_loop(args):

    

    try:
        print(args)
        input()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def main():
    argparser = argparse.ArgumentParser(
        description='Softhand interaction program')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        help='Serial port connect to softhand robot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--title',
        metavar='device_name',
        default='Softhand Robot',
        help='The title to support user')


    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    args = connect_menu(args)

    logging.info('listening to device %s on port %s', args.title, args.port)

    print(__doc__)

    try:
        main_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()

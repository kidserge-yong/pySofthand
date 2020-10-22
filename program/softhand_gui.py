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
    from qbrobot2 import *
    from menu import *
except ImportError:
    raise RuntimeError('cannot import module, make sure sys.path is correct')

# ==============================================================================
# -- control_interface() -------------------------------------------------------
# ==============================================================================

class control_interface():
    def __init__(self, surface, manager, 
        v_offset = 50, h_offset = 40, 
        text_width = 80, 
        button_width = 150, 
        element_height = 50,
        pos = (0, 0)
        ):

        self.surface = surface
        self.manager = manager

        
        self.v_offset = v_offset
        self.h_offset = h_offset
        self.text_width = text_width
        self.button_width = button_width
        self.element_height = element_height

        self.create_ui(self.surface, self.manager, pos)

    def create_ui(self, surface, manager, start_pos = (0,0)):
        self.width, self.height = surface.get_size()

        v_offset, h_offset = self.v_offset, self.h_offset
        status_width, status_height = self.width, self.element_height
        text_width, text_height = self.text_width, self.element_height
        slider_width, slider_height = int(self.width * 0.9), self.element_height
        button_width, button_height = 150, self.element_height
        h_start, v_start = start_pos

        self.status_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(((self.width - status_width)//2 + h_start , v_offset*0+10 + v_start), 
            (status_width, status_height)),
            text = "Initialization",
            manager = manager
            )

        self.max_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(((self.width - slider_width)//2 + slider_width - text_width + h_start, v_offset*2-text_height + v_start), 
            (text_width, text_height)),
            text = "1.0",
            manager = manager
            )

        self.min_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(((self.width - slider_width)//2 + h_start, v_offset*2-text_height + v_start), 
            (text_width, text_height)),
            text = "0.0",
            manager = manager
            )

        self.text_box = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect(((self.width - text_width)//2 + h_start, v_offset*1+20 + v_start), 
            (text_width, text_height)),
            manager = manager
            )
        self.text_box.set_text("0.00")

        self.slider_bar = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect(((self.width - slider_width)//2 + h_start, v_offset*2 + v_start), 
            (slider_width, slider_height)),
            start_value=0,
            value_range=[0.0, 1.0],
            manager=manager
            )

        self.reset_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(((self.width - text_width)//2 + h_start , v_offset*3 + v_start), 
            (text_width, text_height)),
            text = "Reset",
            manager=manager
            )

    def interaction(self, event):
        if event.user_type == pygame_gui.UI_TEXT_ENTRY_FINISHED:
            if event.ui_element == self.text_box:
                try:
                    text_value = float(event.text)
                    if self.slider_bar.value_range[0] <= text_value <= self.slider_bar.value_range[1]:
                        self.slider_bar.set_current_value(text_value)
                        self.status_label.set_text("Set new value")
                        self.input_function(text_value)
                    else:
                        self.status_label.set_text("Outside value range :" + str(self.slider_bar.value_range))
                except ValueError:
                    self.status_label.set_text("Please input number")
                
        elif event.user_type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            if event.ui_element == self.slider_bar:
                self.text_box.set_text("%.2f" % (event.value))
                self.input_function(event.value)
                self.status_label.set_text("Set new value")

        elif event.user_type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.reset_button:
                self.text_box.set_text("%.2f" % (0.0))
                self.slider_bar.set_current_value(0.0)
                self.input_function(0.0)
                self.status_label.set_text("Reset Completed")

    def input_function(self, value):
        self.function(int(value), 0)

    def set_function(self, function):
        self.function = function

    def set_range(self, range_limit):
        self.slider_bar.value_range = range_limit
        self.min_label.set_text(str(range_limit[0]))
        self.max_label.set_text(str(range_limit[1]))
        self.slider_bar.set_current_value(0.0)




# ==============================================================================
# -- main_loop() ---------------------------------------------------------------
# ==============================================================================

def main_loop(args=0):
    """
    Softhand handle
    """
    softhand = robot(args.port)
    softhand.add_part(1, "Hand Grip/Open", "softhand")
    softhand.add_part(2, "Wrist Flex/Exten", "qbmove")
    softhand.add_part(3, "Wrist Pron/Supi", "qbmove")
    """
    gui handle
    """
    width = 600
    height = 600
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    pygame.init()

    pygame.display.set_caption('QBRobot controller')
    surface = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    manager = pygame_gui.UIManager((width, height), 'theme.json')
    
    ci = []
    limit = [
        (0, 19000),
        (-5000, 5000),
        (-5000, 5000)
    ]

    for i in range(3):
        ci.append(control_interface(surface, manager, pos = (0, height/3*i)))
        ci[i].set_function(softhand.get_part(i+1).sendPosStiff)
        ci[i].set_range(limit[i])

    clock = pygame.time.Clock()

    is_running = True

    while is_running:
        time_delta = clock.tick(60)/1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False
            elif event.type == pygame.USEREVENT:
                for item in ci:
                    item.interaction(event)
            elif event.type == pygame.VIDEORESIZE:
                surface = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                manager = pygame_gui.UIManager((event.w, event.h), 'theme.json')
                ci = []
                for i in range(3):
                    ci.append(control_interface(surface, manager, pos = (0, event.h/3*i)))
                    ci[i].set_function(softhand.get_part(i+1).sendPosStiff)
                    ci[i].set_range(limit[i])

            manager.process_events(event)

        manager.update(time_delta)

        
        manager.draw_ui(surface)

        pygame.display.update()

    # try:
    #     print(args)
    #     input()

    # except KeyboardInterrupt:
    #     print('\nCancelled by user. Bye!')


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

    print(str(args))

    logging.info('listening to device %s on port %s', args.title, args.port)

    print(__doc__)

    try:
        main_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
    #main_loop()

import pygame
import pygame_menu

from serial.tools import list_ports
import serial

def connect_menu(args):

    ### initial pygame
    intro_width = 1200   
    intro_height = 600

    pygame.init()
    surface = pygame.display.set_mode((intro_width, intro_height), pygame.RESIZABLE)
    pygame.display.set_caption("Select the correct serial port for Softhand Robot")

    comlist = list_ports.comports()
    serialport = []
    for item in comlist:
        if item:
            serialport.append((str(item), item))

    args.port_name, args.port = serialport[0]
    args.baudrate = 2000000

    def set_serialport(port_name:str, port):
        try:
            args.port_name = str(port)
            args.port = port
        except ValueError:
            print("ERROR: Please enter in type %s" %(type(args.port)))

    def make_a_connection(args):
        try: # test if the connect is possible or not.
            tem = serial.Serial(args.port.device, args.baudrate)
            tem.close()
            args.menu_flag = False
        except serial.SerialException:
            print("ERROR device %s not found on port %s" % (args.port, args.port_name))
        
    args.menu_flag = True

    menu = pygame_menu.Menu(intro_height, intro_width, 'Welcome',
                        theme=pygame_menu.themes.THEME_BLUE)

    menu.add_selector('%s serial port: ' % (args.title), 
                        serialport,
                        default = 0,
                        onchange = set_serialport
                        )
    menu.add_vertical_margin(100)
    menu.add_button('Connect', 
                    make_a_connection,
                    args
                    )
    menu.add_button('Quit', pygame_menu.events.EXIT)

    while args.menu_flag is True:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                exit()
            elif event.type == pygame.VIDEORESIZE:
                surface = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                menu = pygame_menu.Menu(event.h, event.w, 'Welcome',
                        theme=pygame_menu.themes.THEME_BLUE)
                menu.add_selector('%s serial port: ' % (args.title), 
                                    serialport,
                                    default = [x for x,_ in serialport].index(args.port_name),
                                    onchange = set_serialport
                                    )
                menu.add_vertical_margin(100)
                menu.add_button('Connect', 
                                make_a_connection,
                                args
                                )
                menu.add_button('Quit', pygame_menu.events.EXIT)

        menu.update(events)
        menu.draw(surface)

        pygame.display.update()

    return args
import serial
import serial.rs485 as rs485
from service.qbcommand import *
from serial.tools import list_ports
from service.qbdevice import device
import time, sys

import threading
from pylsl import StreamInfo, StreamOutlet

DEBUG = False

from enum import Enum

def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper


class robot():

    is_start = False
    is_lsl = False
    is_update_request = False
    is_new_data = False

    devices = []
    command_buf = []

    pos_outlet = None
    cur_outlet = None

    def __init__(self, serial_port = None, *args, **kwargs):
        self.serial_port = serial.Serial()
        self.pos_outlet = None
        self.cur_outlet = None
        
        self.connect_serial(serial_port)

        super(robot, self).__init__(*args, **kwargs) 
        self._stop = threading.Event() 

    def start(self):
        self.is_start = True
        self.is_update_request = True
        self.mainloop = self.mainloop()

    def stop(self):
        self.is_start = False
        self.is_update_request = False
        del self.mainloop

    @threaded
    def mainloop(self, sleep_interval:float = 0.01): 
        """
        Main loop that process at maximum 100Hz,
        sleep_interval: Because we use separate thread to parallel control qbrobot without block the process, this value also affect sampling frequency of the data
        Task:
        ask position current and stiffness from all device within the robot.
        process return answer from each device and give data to appropriate device.
        update and send data though LSL.
        check command buffer and send all data within command buffer to robot.
        """
        
        print("start mainloop")
        while True:
            time.sleep(sleep_interval)
            if not self.is_start:
                break
            if not self.serial_port.isOpen():
                print("serial_port is not connect to robot.")
                break

            if self.is_update_request is True:
                self.__send_request()

            data_in = self.serial_port.read_all()
            if len(data_in) > 0:
                self.__update_data(data_in)

            if self.is_new_data is True:
                self.update_lsl()
                self.is_new_data = False
            
            #print("in mainloop")
            com_len = len(self.command_buf)
            if com_len > 0:
                for item in self.command_buf:
                    command = self.command_buf.pop(0)
                    self.serial_port.write(command)
        print("break from mainloop")

    def __del__(self):
        """
        deconstrutor to take care of lsl and serial port and ensure it correctly closed.
        """
        if DEBUG:
            print("start deconstrutor")
        if self.is_start is True:
            self.stop()
        if self.is_lsl is True:
            self.stop_lsl()
        if self.serial_port.isOpen():
            self.serial_port.close()
        
    def connect_serial(self, serial_port:serial.tools.list_ports_common.ListPortInfo):
        """
        function taking care of connect robot with serial port.
        serial_port: element of list_ports.comports(), see pyserial for further information
        """
        if serial_port is None:
            print("serial_port is not correct, initial manual serial port selection.")

            while serial_port is None:
                comlist = list_ports.comports()
                id = 0
                for element in comlist:
                    if element:
                        id = id + 1
                        print("ID: " + str(id) + " -- Portname: " + str(element) + "\n")
                port = int(input("Enter a port number: "))
                if port - 1 < len(comlist):
                    serial_port = comlist[port-1]
                else:
                    print("Wrong serial port selected, try again")


            self.openRS485(serial_port)
            #self.add_device(1, "Hand Grip/Open", "softhand")
            #self.add_device(2, "Wrist Flex/Exten", "qbmove")
            #self.add_device(3, "Wrist Pron/Supi", "qbmove")

            for item in self.devices:
                item.activate()
        elif type(serial_port) is serial.tools.list_ports_common.ListPortInfo:
            self.openRS485(serial_port)
        else:
            pass

    def add_device(self, device_id:int = 1, name:str="", dtype:str="softhand"):
        """
        add device according to device_id
        device_id:int, device_id future check communication model before create device will be implement.
        name:str, just a name of device.
        dtype:str, type of device following service.qbcommand.qbrobot_type enum.
        """
        if self.is_update_request:
            print("Please add all device before start robot, stop, del and recreate robot")
            return
        if self.serial_port is None:
            print("Please connect robot to serial port first to confirm the connectivity")
            return
        if self.command_buf is None:
            print("Warning, command buffer do not initialize")

        new_device = device(device_id, name, dtype, self.serial_port, self.command_buf)
        new_device.activate()
        self.devices.append(new_device)
        if self.is_lsl:
            print("Each device need to reconfigurate lsl.")
            self.stop_lsl()
            self.start_lsl()

    def start_lsl(self):
        """
        initial lsl communication and allow data to send according to number of device in the robot
        """
        self.pos_outlet = StreamOutlet(StreamInfo('Softhand Position Data', 'Position', 3 * len(self.devices), 100, 'int16', 'myshpos20191002'))
        self.cur_outlet = StreamOutlet(StreamInfo('Softhand Current Data', 'Current', 2 * len(self.devices), 100, 'int16', 'myshcur20191002'))
        self.is_lsl = True

    def stop_lsl(self):
        """
        recycle lsl communication
        """
        self.pos_outlet = None
        self.cur_outlet = None
        self.is_lsl = False

    def update_lsl(self):
        """
        send new data from each devices though lsl communication.
        """
        if self.is_lsl:
            value = []
            for device_i in self.devices:
                if device_i.pos is not None:
                    value = value + device_i.pos
            if len(value) == (3 * len(self.devices)):
                self.pos_outlet.push_sample(value)
            value = []
            for device_i in self.devices:
                if device_i.cur is not None:
                    value = value + device_i.cur
            if len(value) == (2 * len(self.devices)):
                self.cur_outlet.push_sample(value)

    def __update_data(self, data_in:bytes):
        """
        Private function for send binary data for update each devices.
        data_in: bytes, bytes data that separate by "::"
        """
        datas = data_in.split(str.encode('::'))
        for data in datas:
            if len(data) > 0:
                for device_i in self.devices:
                    if device_i.checkDeviceID(data[0]):
                        device_i.updateData(data)
        self.is_new_data = True

            
    def __send_request(self):
        """
        Private function for add request data for each device in robot.
        """
        if self.is_start == True:
            for device_i in self.devices:
                device_i.getPosition()
                device_i.getCurrent()
        else:
            print("Stop send request")
            return

    def openRS485(self, port):
        """
        open serial port to robot and also configurate the communication protocol according to RS485
        port: serial_port
        """
        self.serial_port = serial.Serial(port.device, BAUD_RATE, timeout=1)
        self.serial_port.rs485_mode = rs485.RS485Settings()
        return 1

    def movedevice(self, device_num:int, position:int = 0, percentStiffness:int = 0):
        """
        move device according to device number: keep for compatibility, will be removed in future
        device_num: int, index of robot device
        position: int, position in integer
        percentStiffness: stiffness in 0 to 100
        """
        if not 0 <= device_num < len(self.devices):
            print("device_num outside device in robot between 0 and %d" % (len(self.devices)))
            return 0
        if not self.devices[device_num].POS_RANGE[0] <= position <= self.devices[device_num].POS_RANGE[1]:
            print("position outside the device range between %d and %d" % (self.devices[device_num].POS_RANGE))
            return 0
        if not 0 <= percentStiffness <= 100:
            print("percentStiffness is between 0 and 100")
            return 0

        stiffrange = self.devices[device_num].STF_RANGE[1] - self.devices[device_num].STF_RANGE[0]
        stiffness = ((percentStiffness / 100) *  stiffrange) + self.devices[device_num].STF_RANGE[0]

        self.devices[device_num].sendPosStiff(int(position), int(stiffness))
        return 1
    
    def get_device(self, device_id:int):
        """
        get the device for outside controller, use with care.
        device_id: id of the device you want to control.
        """
        for item in self.devices:
            if item.device_id == device_id:
                return item
        print("No device id = %d" % (device_id))
        return None
        
if __name__ == "__main__": 
    softhand = robot()
    softhand.add_device(1, "Hand Grip/Open", "softhand")
    softhand.add_device(2, "Wrist Flex/Exten", "qbmove")
    softhand.add_device(3, "Wrist Pron/Supi", "qbmove")
    softhand.start()
    softhand.start_lsl()
    try:
        while(True):
            try:
                print("Control part")
                device_id = int(input("input device id: "))
                position = int(input("input position: "))
                stiffness = int(input("input stiffness: "))
                softhand.movedevice(device_id, position, stiffness)
            except ValueError:
                pass
    except KeyboardInterrupt:
        pass

    softhand.stop_lsl()
    softhand.stop()
    del softhand

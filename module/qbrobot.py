import serial
import serial.rs485 as rs485
from service.qbcommand import *
from serial.tools import list_ports
from service.qbpart import part
import time, sys

from service.repeatedTimer import RepeatedTimer
import threading
from pylsl import StreamInfo, StreamOutlet

DEBUG = False

from enum import Enum

class robot(threading.Thread):

    def __init__(self, serial_port = None, *args, **kwargs):
        self.is_start = True
        self.is_lsl = False
        self.is_update_request = True
        self.new_data = False

        self.parts = []
        self.command_buf = []

        self.serial_port = serial.Serial()
        self.pos_outlet = None
        self.cur_outlet = None
        
        self.connect_serial(serial_port)

        super(robot, self).__init__(*args, **kwargs) 
        self._stop = threading.Event() 
    
    def stop(self): 
        self._stop.set()

    def stopped(self): 
        return self._stop.isSet() 
  
    def run(self): 
        print("start mainloop")
        while True:
            time.sleep(0.01)
            if self.stopped(): 
                return
            if not self.serial_port.isOpen() or not self.is_start:
                return

            if self.is_update_request:
                self.send_request()

            data_in = self.serial_port.read_all()
            if len(data_in) > 0:
                self.update_data(data_in)

            if self.new_data:
                self.update_lsl()
                self.new_data = False
            
            #print("in mainloop")
            com_len = len(self.command_buf)
            if com_len > 0:
                for item in self.command_buf:
                    command = self.command_buf.pop(0)
                    self.serial_port.write(command)

    def __del__(self):
        print("start deconstrutor")
        self.is_start = False
        if self.is_lsl:
            self.stop_lsl()
        if self.serial_port.isOpen():
            self.serial_port.close()
        
    def connect_serial(self, serial_port:serial.tools.list_ports_common.ListPortInfo):
        if type(serial_port) is not serial.tools.list_ports_common.ListPortInfo:
            print("serial_port type is wrong, initial manual serial port selection.")

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
            self.parts.append(part(1, "Hand Grip/Open", "softhand", self.serial_port, self.command_buf))
            self.parts.append(part(2, "Wrist Flex/Exten", "qbmove", self.serial_port, self.command_buf))
            self.parts.append(part(3, "Wrist Pron/Supi", "qbmove", self.serial_port, self.command_buf))

            for item in self.parts:
                self.command_buf.append(item.comActivate(True))
        else:
            self.openRS485(serial_port)

    def add_part(self, device_id:int = 1, name:str="", dtype:str="softhand"):
        self.parts.append(part(device_id, name, dtype, self.serial_port, self.command_buf))
        self.command_buf.append(self.parts[-1].comActivate(True))
        if self.is_lsl:
            print("Each part need to reconfigurate lsl.")
            self.stop_lsl()

    def start_lsl(self):
        self.pos_outlet = StreamOutlet(StreamInfo('Softhand Position Data', 'Position', 3 * len(self.parts), 100, 'int16', 'myshpos20191002'))
        self.cur_outlet = StreamOutlet(StreamInfo('Softhand Current Data', 'Current', 2 * len(self.parts), 100, 'int16', 'myshcur20191002'))
        self.is_lsl = True

    def stop_lsl(self):
        self.pos_outlet = None
        self.cur_outlet = None
        self.is_lsl = False

    def update_lsl(self):
        if self.is_lsl:
            value = []
            for part_i in self.parts:
                if part_i.pos is not None:
                    value = value + part_i.pos
            if len(value) == (3 * len(self.parts)):
                self.pos_outlet.push_sample(value)
            value = []
            for part_i in self.parts:
                if part_i.cur is not None:
                    value = value + part_i.cur
            if len(value) == (2 * len(self.parts)):
                self.cur_outlet.push_sample(value)

    def update_data(self, data_in):
        datas = data_in.split(str.encode(':'))
        for data in datas:
            if len(data) > 0:
                for part_i in self.parts:
                    if part_i.checkDeviceID(data[0]):
                        part_i.updateData(data)
        self.new_data = True

            
    def send_request(self):
        if self.is_start == True:
            for part_i in self.parts:
                self.command_buf.append(part_i.comGetMeasurement())
                self.command_buf.append(part_i.comGetCurrent())
        else:
            print("Stop send request")
            return

    def openRS485(self, port):
        self.serial_port = serial.Serial(port.device,BAUD_RATE,timeout=1)
        self.serial_port.rs485_mode = rs485.RS485Settings()
        return 1

    def movePart(self, part_num:int, position:int = 0, percentStiffness:int = 0):
        """
        move part according to part number: keep for compatibility, will be removed in future
        part_num: int, index of robot part
        position: int, position in integer
        percentStiffness: stiffness in 0 to 100
        """

        relativePosition = position + self.parts[part_num].pos_offset

        stiffrange = self.parts[part_num].MAX_STF - self.parts[part_num].MIN_STF
        stiffness = ((percentStiffness / 100) *  stiffrange) + self.parts[part_num].MIN_STF

        self.parts[part_num].sendPosStiff(int(relativePosition), int(stiffness))
        return 1
    
    def get_part(self, part_id:int):
        for item in self.parts:
            if item.device_id == part_id:
                return item
        print("No part id = %d" % (part_id))
        return None
        
if __name__ == "__main__": 
    pass
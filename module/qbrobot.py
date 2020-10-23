import serial
import serial.rs485 as rs485
from service.qbcommand import *
from serial.tools import list_ports
from service.qbpart import part
import time

from service.repeatedTimer import RepeatedTimer
from threading import Thread
from pylsl import StreamInfo, StreamOutlet

DEBUG = False

def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper

from enum import Enum

class robot():

    def __init__(self, serial_port:serial.tools.list_ports_common.ListPortInfo = None):
        self.is_lsl = False
        self.is_start = False

        self.parts = []

        self.serial_port = None
        self.update_timer = None
        self.lsl_timer = None
        self.pos_outlet = None
        self.cur_outlet = None
        
        self.handle_receive = []
        self.handle_get = []
        if serial_port is not None:
            self.connect_serial(serial_port)
        

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
            self.parts.append(part(1, "Hand Grip/Open", "softhand", self.serial_port))
            self.parts.append(part(2, "Wrist Flex/Exten", "qbmove",self.serial_port))
            self.parts.append(part(3, "Wrist Pron/Supi", "qbmove", self.serial_port))

            for item in self.parts:
                self.serial_port.write(item.comActivate(True))
        else:
            self.openRS485(serial_port)



    def add_part(self, device_id:int = 1, name:str="", dtype:str="softhand"):
        self.parts.append(part(device_id, name, dtype, self.serial_port))
        self.serial_port.write(self.parts[-1].comActivate(True))
        if self.is_lsl:
            self.stop_lsl()
        

    def start_lsl(self, interval = 0.01):
        self.pos_outlet = StreamOutlet(StreamInfo('Softhand Position Data', 'Position', 3 * len(self.parts), 100, 'int16', 'myshpos20191002'))
        self.cur_outlet = StreamOutlet(StreamInfo('Softhand Current Data', 'Current', 2 * len(self.parts), 100, 'int16', 'myshcur20191002'))
        self.lsl_timer = RepeatedTimer(interval, self.lslloop)
        self.lsl_timer.start()
        self.is_lsl = True

    def stop_lsl(self):
        self.pos_outlet = None
        self.cur_outlet = None
        if self.lsl_timer is not None:
            self.lsl_timer.stop()
            self.lsl_timer = None
        self.is_lsl = False

    def start_updateloop(self, interval = 0.1):
        self.is_start = True
        self.update_timer = RepeatedTimer(interval, self.updateloop)
        self.update_timer.start()
        self.handle_receive = self.getdata_threaded()


    def stop_updateloop(self):
        self.is_start = False
        self.update_timer.stop()
        self.update_timer = None
        self.handle_receive.daemon()
        self.handle_receive = None

    @threaded
    def getdata_threaded(self):
        print("start loop for receive data")
        while True:
            if not self.serial_port.isOpen():
                return
            data_in = self.serial_port.read_all()
            if len(data_in) > 0:
                datas = data_in.split(str.encode(':'))
                for data in datas:
                    if len(data) > 0:
                        for part_i in self.parts:
                            if part_i.checkDeviceID(data[0]):
                                part_i.updateData(data)

    def lslloop(self):
        if not self.is_start: 
            self.lsl_timer.stop()
            return

        if self.is_lsl:
            value = []
            for part_i in self.parts:
                value = value + part_i.pos
            if len(value) == (3 * len(self.parts)):
                self.pos_outlet.push_sample(value)
            value = []
            for part_i in self.parts:
                value = value + part_i.cur
            if len(value) == (2 * len(self.parts)):
                self.cur_outlet.push_sample(value)

    def updateloop(self):
        if self.is_start == True:
            for part_i in self.parts:
                # if part_i.new_pos_target:
                #     part_i.new_pos_target = False
                #     self.serial_port.write(part_i.comSetPosStiff(part_i.pos_target, part_i.sti))
                # if part_i.new_cur_target:
                #     part_i.new_cur_target = False
                #     self.serial_port.write(part_i.comSetPosStiff(part_i.pos_target, part_i.sti))
                self.serial_port.write(part_i.comGetMeasurement())
                self.serial_port.write(part_i.comGetCurrent())
        else:
            print("Stop update data")
            self.update_timer.stop()
            return

    def __del__(self):
        self.is_start = False
        print("Try to stop loop function.")
        time.sleep(2.0)
        self.stop_lsl()
        if self.update_timer is not None:
            self.update_timer.stop()
        if self.serial_port.isOpen():
            self.serial_port.close()

    def openRS485(self, port):
        self.serial_port = serial.Serial(port.device,BAUD_RATE,timeout=1)
        self.serial_port.rs485_mode = rs485.RS485Settings()
        return 1

    def movePart(self, part_num:int, position:int = 0, percentStiffness:int = 0):

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
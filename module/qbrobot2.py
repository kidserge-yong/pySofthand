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

class Qbrobot():

    ROBOT_NAME = "Qbrobot"

    @staticmethod
    def qbrobot_init(data):
        data[ROBOT_NAME] = 

class robot2():
    TIME_CYCLE = 0.01
    TIME_OUT = 30 * 30 # 900 second for 15 minutes without communication

    part = []



class robot():
    ser = serial.Serial()
    lsl = False
    start = False

    part = []

    update_timer = None
    lsl_timer = None
    pos_outlet = None
    cur_outlet = None
    
    handle_receive = []35
    handle_get = []

    def __init__(self, auto_port = True):
        if auto_port:
            self.openRS485(portlist())
            self.part.append(part(1, "Hand Grip/Open", "softhand", self.ser))
            self.part.append(part(2, "Wrist Flex/Exten", "qbmove",self.ser))
            self.part.append(part(3, "Wrist Pron/Supi", "qbmove", self.ser))

            self.ser.write(self.part[0].comActivate(True))
            self.ser.write(self.part[1].comActivate(True))
            self.ser.write(self.part[2].comActivate(True))

    def start_lsl(self, interval = 0.01):
        self.pos_outlet = StreamOutlet(StreamInfo('Softhand Position Data', 'Position', 3 * len(self.part), 100, 'int16', 'myshpos20191002'))
        self.cur_outlet = StreamOutlet(StreamInfo('Softhand Current Data', 'Current', 2 * len(self.part), 100, 'int16', 'myshcur20191002'))
        self.lsl_timer = RepeatedTimer(interval, self.lslloop)
        self.lsl_timer.start()
        self.lsl = True

    def stop_lsl(self):
        self.pos_outlet = None
        self.cur_outlet = None
        if self.lsl_timer is not None:
            self.lsl_timer.stop()
            self.lsl_timer = None
        self.lsl = False

    def start_updateloop(self, interval = 0.1):
        self.start = True
        self.update_timer = RepeatedTimer(interval, self.updateloop)
        self.update_timer.start()
        self.handle_receive = self.getdata_threaded()


    def stop_updateloop(self):
        self.start = False
        self.update_timer.stop()
        self.update_timer = None
        self.handle_receive.daemon()
        self.handle_receive = None

    @threaded
    def getdata_threaded(self):
        print("start loop for receive data")
        while True:
            if not self.ser.isOpen():
                return
            data_in = self.ser.read_all()
            if len(data_in) > 0:
                datas = data_in.split(str.encode(':'))
                for data in datas:
                    if len(data) > 0:
                        for part_i in self.part:
                            if part_i.checkDeviceID(data[0]):
                                part_i.updateData(data)

    def lslloop(self):
        if not self.start: 
            self.lsl_timer.stop()
            return

        if self.lsl:
            value = []
            for part_i in self.part:
                value = value + part_i.pos
            if len(value) == (3 * len(self.part)):
                self.pos_outlet.push_sample(value)
            value = []
            for part_i in self.part:
                value = value + part_i.cur
            if len(value) == (2 * len(self.part)):
                self.cur_outlet.push_sample(value)

    def updateloop(self):
        if self.start == True:
            for part_i in self.part:
                # if part_i.new_pos_target:
                #     part_i.new_pos_target = False
                #     self.ser.write(part_i.comSetPosStiff(part_i.pos_target, part_i.sti))
                # if part_i.new_cur_target:
                #     part_i.new_cur_target = False
                #     self.ser.write(part_i.comSetPosStiff(part_i.pos_target, part_i.sti))
                self.ser.write(part_i.comGetMeasurement())
                self.ser.write(part_i.comGetCurrent())
        else:
            print("Stop update data")
            self.update_timer.stop()
            return

    def __del__(self):
        self.start = False
        print("Try to stop loop function.")
        time.sleep(2.0)
        self.stop_lsl()
        if self.update_timer is not None:
            self.update_timer.stop()
        if self.ser.isOpen():
            self.ser.close()

    def openRS485(self, port):
        self.ser = serial.Serial(port.device,2000000,timeout=1)
        self.ser.rs485_mode = rs485.RS485Settings()
        return 1

    def movePart(self, part_num:int, position:int = 0, percentStiffness:int = 0):

        relativePosition = position + self.part[part_num].pos_offset

        stiffrange = self.part[part_num].MAX_STF - self.part[part_num].MIN_STF
        stiffness = ((percentStiffness / 100) *  stiffrange) + self.part[part_num].MIN_STF

        self.part[part_num].sendPosStiff(int(relativePosition), int(stiffness))
        return 1
        
if __name__ == "__main__": 
    pass
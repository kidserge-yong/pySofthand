import serial
import serial.rs485 as rs485
from serial.tools import list_ports
from service.smkcommand import smkcommandv2 as smkcommand
import time
from threading import Thread
import csv

from pylsl import StreamInfo, StreamOutlet

DEBUG = False


def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper


class smk_arrayemg():
    port = serial.Serial()
    emg = [0 for _ in range(smkcommand.CHANNEL_NUM.value)]
    offset = [-1 for _ in range(smkcommand.CHANNEL_NUM.value)]

    version = smkcommand.VERSION.value

    start = False
    new_data = False
    lsl = False

    handle = None
    outlet = None

    utility = []

    def __init__(self):
        self.opensmk(self.portlist())
        #self.start_data(False)
        #self.stop_data()
        #print(self.emg)
        #if self.emg == [0 for _ in range(smkcommand.CHANNEL_NUM.value)]:
        #    print("Post connection unsuccessful, check connection and serial port number.")

    def __del__(self):
        if DEBUG:
            print("start deconstruct smk_arrayemg")
        
        self.stop_data()
        time.sleep(1.000)
        self.port.close()


    def start_lsl(self):
        self.outlet = StreamOutlet(StreamInfo('SMK Array EMG system', 'EMG', 32, 100, 'int16', 'mysmk20191002'))
        self.lsl = True

    def stop_lsl(self):
        self.outlet = None
        self.lsl = False

    def portlist(self):
        comlist = list_ports.comports()
        id = 0
        for element in comlist:
            if element:
                id = id + 1
                print("ID: " + str(id) + " -- Portname: " + str(element) + "\n")
        port = int(input("Enter a port number: "))
        if DEBUG:
            print("You select " + str(comlist[port-1]))
        return comlist[port-1]

    def opensmk(self, port):
        self.port = serial.Serial(port.device, smkcommand.BAUDRATE.value, timeout=1)
        return 1

    @threaded
    def getdata_threaded(self):
        print("start loop for data in SMK")
        while True:
            if self.start and self.port.isOpen():
                self.getdata()
            else:
                print("no start")
                if self.start:
                    self.stop_data()
                return -1

    def start_data(self, START_LOOP = True):
        if not self.port.isOpen():
            return -1

        if self.version == 1:
            command = bytes([smkcommand.CMD_START.value])
        elif self.version == 2:
            command = smkcommand.CMD_START.value.encode()

        self.port.write(command)

        # wait for response
        time.sleep(.200)
        self.getdata()
        self.getdata()
        self.offset = [x for x in self.emg]

        # Start receive loop
        self.start = True
        if START_LOOP:
            self.handle = self.getdata_threaded()

    def stop_data(self):
        if not self.port.isOpen():
            return -1

        if self.version == 1:
            command = bytes([smkcommand.CMD_STOP.value])
        elif self.version == 2:
            command = smkcommand.CMD_STOP.value.encode()

        self.port.write(command)

        print("stop get data")

        # Stop receive loop
        self.start = False
        time.sleep(.200)
        self.handle = None

    def calemg(self):
        emg = [x - 65536 if x > 32767 else x for x in self.emg]
        offset = [y - 65536 if y > 32767 else y for y in self.offset]
        self.new_data = False
        return [j-k for j,k in zip(emg,offset)]

    def formatOutput(self):
        output = self.calemg()
        text = ''
        for element in output:
            text += '{0}'.format(element) + '\t'
        return text


    def getdata(self):
        # Check state of system
        if not self.start:
            return "System is not started and getdata was called"
        if not self.port.isOpen():
            return "Port is not connected and getdata was called"

        
        # Check version of system
        try:
            if self.version == 1:
                self.getdata_v1()
            elif self.version == 2:
                self.getdata_v2()
        except:
            pass

    def getdata_v1(self):
        receive = self.port.read(1)
        if len(receive == 0):
            return

        if receive[0] == 113:
            receive = self.port.read(17)
            data_byte = receive[1:]
            value = []
            for i, k in zip(data_byte[0::2], data_byte[1::2]):
                value.append((k * 256) + i)
            self.emg[0:8] = value[0:8]
        elif receive[0] == 114:
            receive = self.port.read(17)
            data_byte = receive[1:]
            value = []
            for i, k in zip(data_byte[0::2], data_byte[1::2]):
                value.append((k * 256) + i)
            self.emg[8:12] = value[0:4]
            self.new_data = True                        #only after this all data will be new data
            if self.lsl:
                self.outlet.push_sample(self.emg)
        else:
            print("first byte error not 113 or 114")

    def getdata_v2(self):

        value = []
        trigger = []
        check = ord(self.port.read(1))
        #print(check)
        while not (check == smkcommand.CMD_CHECK_IEMG.value or check == smkcommand.CMD_CHECK_EMG.value):
            check = ord(self.port.read(1))
            #print(check)

        data_byte = self.port.read(68)


        trigger = data_byte[67:]
        for i, k in zip(data_byte[1:67:2], data_byte[2:67:2]):
            value.append((k * 256) + i)
        self.emg[0:32] = value[1:33]
        self.new_data = True
        if self.lsl:
            self.outlet.push_sample(self.emg)


        utility = [value[0]] + trigger



    def calibrate(self, channel=0):
        if not self.port.isOpen():
            return "Port is not connected and calibrate was called"
        if not self.version == 1:
            return "Do not have calibrate command for this version"

        if channel == 0:
            command = bytes([smkcommand.CMD_START_ALL_CALIBRATE.value])
        else:
            command = bytes([smkcommand.CMD_START_ONE_CALIBRATE.value])
            command.append(bytes([channel]))
        self.port.write(command)
        print("Wait 1 second")
        time.sleep(1.000)
        command = bytes([smkcommand.CMD_STOP_CALIBRATE.value])
        self.port.write(command)

    def save2File(self, output, filename = 'SMKoutput.csv'):
        with open(filename, mode='w', newline='') as outputfile:
            output_writer = csv.writer(outputfile)
            output_writer.writerows(output)

    def read4File(self, filename = 'SMKoutput.csv'):
        data = []
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                try:
                    data.append([int(value) for value in row])
                except:
                    pass
        return data


if __name__ == "__main__": 
    smk = smk_arrayemg()
    smk.start_data()
    smk.start_lsl()

    time.sleep(1.000)
    record_time = 5.000 # record for 5s

    t_end = time.time() + record_time
    emg = []
    try:
        print("Use Ctrl + C to emergency stop recording.")
        while time.time() < t_end:
            if smk.new_data:
                emg.append(smk.calemg())
        print("Finish rest state.")
    except KeyboardInterrupt:
        pass

    smk.stop_lsl()
    smk.stop_data()
    smk.save2File(emg)
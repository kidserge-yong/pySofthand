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
    serial_port = serial.Serial()
    __emg = [0 for _ in range(smkcommand.CHANNEL_NUM.value)]
    __offset = [-1 for _ in range(smkcommand.CHANNEL_NUM.value)]

    version = "0.0.0.0"

    is_loop = True
    is_start = False
    is_new_data = False
    is_lsl = False

    __handle = None
    __outlet = None

    command_buf = []
    utility = []

    def __init__(self, serial_port = None, *args, **kwargs):
        
        try: 
            self.connect_serial(serial_port)
        except:
            print("connection fail, please manualy connect to serial port")

        self.restart()

    def restart(self):
        self.__emg = [0 for _ in range(smkcommand.CHANNEL_NUM.value)]
        self.__offset = [-1 for _ in range(smkcommand.CHANNEL_NUM.value)]

        self.version = smkcommand.VERSION.value

        self.is_loop = True
        self.is_start = False
        self.is_new_data = False
        self.is_lsl = False

        self.__handle = None
        self.__outlet = None

        self.command_buf = []
        self.utility = []

        self.__handle = self.mainloop()

    def start(self):
        if not self.serial_port.isOpen():
            print("Please connect to serial port before start communication")
            return 0

        self.__start_data()
        self.is_start = True
        return 1

    def stop(self):
        if not self.serial_port.isOpen():
            print("Please connect to serial port before start communication")
            return 0

        self.__stop_data()
        self.is_loop = False
        self.is_start = False
        return 1
    
    @threaded
    def mainloop(self, sleep_interval:float = 0.01):
        print("start loop for data in SMK")
        while True:
            time.sleep(sleep_interval)
            if not self.serial_port.isOpen():
                print("serial_port is disconnected from robot.")
                break
            if not self.is_loop:
                break
            
            if self.is_start:
                self.__getdata()
            if self.is_lsl and self.is_new_data:
                self.__outlet.push_sample(self.__emg)

            com_len = len(self.command_buf)
            if com_len > 0:
                for item in self.command_buf:
                    command = self.command_buf.pop(0)
                    self.serial_port.write(command)
        print("break from mainloop")

    def __del__(self):
        if DEBUG:
            print("start deconstruct smk_arrayemg")
        
        self.stop()
        time.sleep(1.000)
        self.serial_port.close()


    def start_lsl(self):
        self.__outlet = StreamOutlet(StreamInfo('SMK Array EMG system', 'EMG', 32, 100, 'int16', 'mysmk20191002'))
        self.is_lsl = True

    def stop_lsl(self):
        self.__outlet = None
        self.is_lsl = False

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
                print(serial_port)


            self.__opensmk(serial_port)
        elif type(serial_port) is serial.tools.list_ports_common.ListPortInfo:
            self.__opensmk(serial_port)
        else:
            pass

    def __opensmk(self, serial_port):
        if DEBUG:
            print("initial port communcation with %s" % (serial_port))
        self.serial_port = serial.Serial(serial_port.device, smkcommand.BAUDRATE.value, timeout=1)
        return 1

    def __start_data(self):
        if not self.serial_port.isOpen():
            return 0

        if self.version == 1:
            command = bytes([smkcommand.CMD_START.value])
        elif self.version == 2:
            command = smkcommand.CMD_START.value.encode()

        self.command_buf.append(command)

        # wait for response
        time.sleep(.200)
        self.__getdata()
        self.__getdata()
        self.__offset = [x for x in self.__emg]

    def __stop_data(self):
        if not self.serial_port.isOpen():
            return 0

        if self.version == 1:
            command = bytes([smkcommand.CMD_STOP.value])
        elif self.version == 2:
            command = smkcommand.CMD_STOP.value.encode()

        self.command_buf.append(command)

    def emg(self):
        emg = [x - 65536 if x > 32767 else x for x in self.__emg]
        offset = [y - 65536 if y > 32767 else y for y in self.__offset]
        self.is_new_data = False
        return [j-k for j,k in zip(emg,offset)]

    def formatOutput(self):
        output = self.emg()
        text = ''
        for element in output:
            text += '{0}'.format(element) + '\t'
        return text


    def __getdata(self):
        # Check state of system
        if not self.is_start:
            return "System is not started and getdata was called"
        if not self.serial_port.isOpen():
            return "Port is not connected and getdata was called"

        
        # Check version of system
        try:
            if self.version == 1:
                self.__getdata_v1()
            elif self.version == 2:
                self.__getdata_v2()
        except:
            pass

    def __getdata_v1(self):
        receive = self.serial_port.read(1)
        if len(receive == 0):
            return

        if receive[0] == 113:
            receive = self.serial_port.read(17)
            data_byte = receive[1:]
            value = []
            for i, k in zip(data_byte[0::2], data_byte[1::2]):
                value.append((k * 256) + i)
            self.__emg[0:8] = value[0:8]
        elif receive[0] == 114:
            receive = self.serial_port.read(17)
            data_byte = receive[1:]
            value = []
            for i, k in zip(data_byte[0::2], data_byte[1::2]):
                value.append((k * 256) + i)
            self.__emg[8:12] = value[0:4]
            self.is_new_data = True                        #only after this all data will be new data
        else:
            print("first byte error not 113 or 114")

    def __getdata_v2(self):

        value = []
        trigger = []
        check = ord(self.serial_port.read(1))
        #print(check)
        while not (check == smkcommand.CMD_CHECK_IEMG.value or check == smkcommand.CMD_CHECK_EMG.value):
            check = ord(self.serial_port.read(1))
            #print(check)

        data_byte = self.serial_port.read(68)


        trigger = data_byte[67:]
        for i, k in zip(data_byte[1:67:2], data_byte[2:67:2]):
            value.append((k * 256) + i)
        self.__emg[0:32] = value[1:33]
        self.is_new_data = True


        utility = [value[0]] + trigger



    def calibrate(self, channel=0):
        if not self.serial_port.isOpen():
            return "Port is not connected and calibrate was called"
        if not self.version == 1:
            return "Do not have calibrate command for version" % (self.version)

        if channel == 0:
            command = bytes([smkcommand.CMD_START_ALL_CALIBRATE.value])
        else:
            command = bytes([smkcommand.CMD_START_ONE_CALIBRATE.value])
            command.append(bytes([channel]))
        self.command_buf.append(command)
        print("Wait 1 second")
        time.sleep(1.000)
        command = bytes([smkcommand.CMD_STOP_CALIBRATE.value])
        self.command_buf.append(command)

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
    smk.start()
    smk.start_lsl()

    time.sleep(1.000)
    record_time = 5.000 # record for 5s

    t_end = time.time() + record_time
    emg = []
    try:
        print("Use Ctrl + C to emergency stop recording.")
        while time.time() < t_end:
            if smk.is_new_data:
                emg.append(smk.emg())
        print("Finish rest state.")
    except KeyboardInterrupt:
        pass

    smk.stop_lsl()
    smk.stop()
    smk.save2File(emg)
import serial
import serial.rs485 as rs485
from qbcommand import *
from serial.tools import list_ports
import time

DEBUG = False


class part():
    device_id = 1
    device_name = ""
    device_type = "softhand"    # {softhand, qbmove}

    new_pos = False
    new_cur = False
    new_pos_target = False
    new_cur_target = False

    pos_target = 0
    cur_target = 0
    pos = 0
    cur = 0
    stf = 0
    pos_offset = 0
    cur_offset = 0
    stf_offset = 0

    serial_port = None

    MAX_POS = 32767
    MIN_POS = -32766
    MAX_CUR = 32767
    MIN_CUR = -32766
    MAX_STF = 32767
    MIN_STF = 0

    def __init__(self, new_id=1, name="", dtype="softhand",serial=None):
        self.device_id = new_id
        self.device_name = name
        self.device_type = dtype

        if serial is None:
            return

        self.serial_port = serial
        self.checkConnectivity()
        self.sendSerial(self.comActivate(False))        ## deactivate
        
    def comActivate(self, activate:bool=True):
        command = qbmove_command.CMD_ACTIVATE
        data = []
        if activate:
            data.append(3)
        else:
            data.append(0)
        return self.comContrustion(command, True, data)

        
    def sendPosStiff(self, position=0, stiff=0):
        if self.serial_port.isOpen():
            if self.device_type is "softhand":
                command = self.comSetPosition(position)
                self.serial_port.write(command)
            elif self.device_type is "qbmove":
                command = self.comSetPosStiff(position, stiff)
                self.serial_port.write(command)
            return command
        return 0

    def setTargetPosition(self, t_position):
        if t_position > self.MAX_POS or t_position < self.MIN_POS:
            print("Device %d: Wrong target position" % self.device_id)
            return
        self.pos_target = t_position
        self.new_pos_target = True

    def setTargetCurrent(self, t_current):
        if t_current > self.MAX_CUR or t_current < self.MIN_CUR:
            print("Device %d: Wrong target current" % self.device_id)
            return
        self.cur_target = t_current
        self.new_cur_target = True

    def checkDeviceID(self, ID):
        if isinstance(ID, int):
            return self.device_id == ID
        elif isinstance(ID, str):
            return self.device_name == ID
        return False

    def comPing(self):
        com = self.comContrustion(qbmove_command.CMD_PING)
        return(com)

    def comGetMeasurement(self):
        com = self.comContrustion(qbmove_command.CMD_GET_MEASUREMENTS)
        return(com)

    def comGetCurrent(self):
        com = self.comContrustion(qbmove_command.CMD_GET_CURRENTS)
        return(com)

    def comSetPosition(self, position):
        if position > self.MAX_POS or position < self.MIN_POS:
            print("Position should be between " + str(self.MAX_POS) +
                  " to " + str(self.MIN_POS) + " (2 bytes with signed)")
            return

        command = qbmove_command.CMD_SET_INPUTS
        data = []
        data.append(position.to_bytes(4, byteorder='little', signed=True)[1])
        data.append(position.to_bytes(4, byteorder='little', signed=True)[0])
        data.append(position.to_bytes(4, byteorder='little', signed=True)[3])
        data.append(position.to_bytes(4, byteorder='little', signed=True)[2])
        return self.comContrustion(command, True, data)

    def comSetPosStiff(self, position, stiff):
        if position > self.MAX_POS or position < self.MIN_POS:
            print("Position should be between " + str(self.MAX_POS) +
                  " to " + str(self.MIN_POS) + " (2 bytes with signed)")
            return
        if stiff > self.MAX_STF or stiff < self.MIN_STF:
            print("Stiffness should be between" + str(self.MAX_STF) +
                  " to " + str(self.MIN_STF) + " (2 bytes with signed)")
            return

        command = qbmove_command.CMD_SET_POS_STIFF
        data = []
        data.append(position.to_bytes(2, byteorder='little', signed=True)[1])
        data.append(position.to_bytes(2, byteorder='little', signed=True)[0])
        data.append(stiff.to_bytes(2, byteorder='little', signed=True)[1])
        data.append(stiff.to_bytes(2, byteorder='little', signed=True)[0])
        return self.comContrustion(command, True, data)

    def comHandCalibrate(self, speed, repetitions):
        print("Speed between 0-200 and repetitions between 0-32767")
        command = qbmove_command.CMD_HAND_CALIBRATE
        data = []
        data.append(speed.to_bytes(2, byteorder='little')[1])
        data.append(speed.to_bytes(2, byteorder='little')[0])
        data.append(repetitions.to_bytes(2, byteorder='little')[1])
        data.append(repetitions.to_bytes(2, byteorder='little')[0])
        return self.comContrustion(command, True, data)

    def updateData(self, data):
        # data = data[2:]
        data = data.replace(str.encode(':'),b'')
        # print(data)
        if len(data) > 3:

            if data[0] != self.device_id:
                return

            if self.byte2int(self.checksum(data[2:])) == 0:  # checksum
                value = []
                data_value = data[3:-1]
                info = [data_value[i:i+2]
                        for i in range(0, len(data_value), 2)]
                for i in info:
                    value.append(int.from_bytes(
                        i, byteorder='big', signed=True))
                # for i,k in zip(data[3:-1:2], data[4:-1:2]):
                #     value.append((i * 256) + k)
                if 128 <= data[2] <= 146:
                    command = qbmove_command(data[2])  # data type
                else:
                    print(data[2])
                    return
                if command == qbmove_command.CMD_GET_MEASUREMENTS:
                    self.pos = value
                    self.new_pos = True
                elif command == qbmove_command.CMD_GET_CURRENTS:
                    self.cur = value
                    self.new_cur = True
            else:
                if DEBUG:
                    print(self.checksum(data[2:]))

    def __str__(self):
        return "ID: %d\nPosition: %d\Current: %d" % (self.device_id, self.pos, self.cur)

    def getinfo(self):
        c1 = str.encode('?')
        c2 = bytes([13])
        c3 = bytes([10])
        return(c1+c2+c3)

    def comContrustion(self, command, checksum=False, data=None):
        command_list = []
        command_list.append(command.value)
        data_byte = bytes()

        if data is None:
            command_list.append(command.value)
            length = len(command_list)
        else:
            length = len(command_list) + len(data)
            for data_i in data:
                data_byte = data_byte + bytes([data_i])

        # convert all to byte data

        start_byte = str.encode('::')
        device_id = bytes([self.device_id])
        if checksum:
            length_byte = bytes([length + 1])
        else:
            length_byte = bytes([length])
        command_byte = bytes()
        for command_i in command_list:
            command_byte = command_byte + bytes([command_i])

        # combine data and add checksum

        com = start_byte+device_id+length_byte+command_byte+data_byte

        if checksum:
            com = com + self.checksum(com[4:])

        if DEBUG:
            print("Command is : ")
            print(com)

        return(com)

    def sendSerial(self, data):
        if self.serial_port is not None:
            self.serial_port.write(data)
        else:
            if DEBUG:
                print("serial port donot connect with part ID:%d" %
                      (self.device_id))

    def checksum(self, data_buffer, data_length=0):
        checksum = bytes([0])
        buffer = data_buffer
        if data_length == 0:
            data_length = len(buffer)
        for data in buffer:
            checksum = self.byte_xor(checksum, bytes([data]))

        return checksum

    def byte_xor(self, ba1, ba2):
        return bytes([_a ^ _b for _a, _b in zip(ba1, ba2)])

    def byte2int(self, data_bytes):
        return int.from_bytes(data_bytes, byteorder='little', signed=True)

    def checkConnectivity(self):
        print("Test Connection of qbrobot:part%d. Send ping command" %
              (self.device_id))
        com = self.comPing()

        if DEBUG:
            print("command is %s" % com)

        self.serial_port.read_all()  # Delete all data in buffer
        self.sendSerial(com)
        print("Wait for response 1.5 second.")
        
        time.sleep(1.5)

        check = self.serial_port.read_all()

        if DEBUG:
            print("check is %s" % check)

        if check != com:
            print("Connection test fail, please check the serial port connectivity. Drop serial port")
            self.serial_port = None
            return
        
        print("Connection confirm with return: %s" % (com))

        while self.pos == 0:
            print("Start receive first data")
            self.serial_port.read_all()  # Delete all data in buffer
            self.sendSerial(self.comGetMeasurement())
            print("Wait for response 1.5 second.")
            
            time.sleep(1.5)
            data = self.serial_port.read_all()
            if DEBUG:
                print("Raw Position is %s" % data)
            self.updateData(data)
            print("Position: %s \nCurrent: %s" % (self.pos, self.cur))
        
        self.pos_offset = self.pos[1]
        
        
        while self.cur == 0:
            self.serial_port.read_all()  # Delete all data in buffer
            self.sendSerial(self.comGetCurrent())
            
            print("Wait for response 1.5 second.")
            
            time.sleep(1.5)
            data = self.serial_port.read_all()
            if DEBUG:
                print("Raw Current is %s" % data)
            self.updateData(data)
            print("Position: %s \nCurrent: %s" % (self.pos, self.cur))
        
        self.cur_offset = self.cur

        print("Finish test for connectivity of qbrobot:part%d." %
              (self.device_id))

import serial
import serial.rs485 as rs485
from service.qbcommand import *
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

    pos_target = None
    cur_target = None
    pos = None
    cur = None
    stf = None
    pos_offset = None
    cur_offset = None
    stf_offset = None

    serial_port = None

    POS_RANGE = (-32766, 32767)
    CUR_RANGE = (-32766, 32767)
    STF_RANGE = (0, 32767)
    MINIMAL_CHANGES = 100

    def __init__(self, new_id=1, name="", dtype="softhand", serial=None, buffer = None):
        if serial is None:
            return

        self.buffer = buffer
        
        self.device_id = new_id
        self.device_name = name
        self.device_type = dtype

        if dtype == qbrobot_type.SOFTHAND.name.lower():
            self.POS_RANGE = (0, 19000)
        elif dtype == qbrobot_type.QBMOVE.name.lower():
            self.POS_RANGE = (-32766, 32767)

        self.serial_port = serial
        self.is_connect = self.checkConnectivity()
        if self.is_connect is False:
            return
        self.sendSerial(self.comActivate(False))        ## deactivate
        self.pos_target = 0
        self.cur_target = 0
    
    def get_range(self):
        return [self.POS_RANGE, self.CUR_RANGE, self.STF_RANGE]

    def set_range(self, range_limit:list = []):
        if len(range) == 3:
            self.POS_RANGE = range_limit[0]
            self.CUR_RANGE = range_limit[1]
            self.STF_RANGE = range_limit[2]
            return 1
        else:
            print("range_limit in wrong format")
            return 0

        
    def comActivate(self, activate:bool=True):
        command = qbmove_command.CMD_ACTIVATE
        data = []
        if activate:
            data.append(3)
        else:
            data.append(0)
        return self.comContrustion(command, True, data)
        
    def sendPosStiff(self, position=0, stiff=0):
        if not self.POS_RANGE[0] <= position <= self.POS_RANGE[1]:
            print("Device %d: Wrong target position at position %d" % (self.device_id, position))
            return 0
        if self.serial_port.isOpen() is False:
            return 0

        if abs(position - self.pos[0]) > self.MINIMAL_CHANGES:
            self.pos_target = position
            if self.device_type is "softhand":
                command = self.comSetPosition(position)
                self.buffer.append(command)
            elif self.device_type is "qbmove":
                command = self.comSetPosStiff(position, stiff)
                self.buffer.append(command)
            return command
        else:
            return 0
    
    def setTargetPosition(self, t_position):
        if t_position > self.POS_RANGE[1] or t_position < self.POS_RANGE[0]:
            print("Device %d: Wrong target position" % self.device_id)
            return
        self.pos_target = t_position
        self.new_pos_target = True

    def setTargetCurrent(self, t_current):
        if t_current > self.CUR_RANGE[1] or t_current < self.CUR_RANGE[0]:
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
        if position > self.POS_RANGE[1] or position < self.POS_RANGE[0]:
            print("Position should be between " + str(self.POS_RANGE[1]) +
                  " to " + str(self.POS_RANGE[0]) + " (2 bytes with signed)")
            return

        command = qbmove_command.CMD_SET_INPUTS
        data = []
        data.append(position.to_bytes(4, byteorder='little', signed=True)[1])
        data.append(position.to_bytes(4, byteorder='little', signed=True)[0])
        data.append(position.to_bytes(4, byteorder='little', signed=True)[3])
        data.append(position.to_bytes(4, byteorder='little', signed=True)[2])
        return self.comContrustion(command, True, data)

    def comSetPosStiff(self, position, stiff):
        if position > self.POS_RANGE[1] or position < self.POS_RANGE[0]:
            print("Position should be between " + str(self.POS_RANGE[1]) +
                  " to " + str(self.POS_RANGE[0]) + " (2 bytes with signed)")
            return
        if stiff > self.STF_RANGE[1] or stiff < self.STF_RANGE[0]:
            print("Stiffness should be between" + str(self.STF_RANGE[1]) +
                  " to " + str(self.STF_RANGE[0]) + " (2 bytes with signed)")
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

    def cominfo(self):
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
            #self.buffer.append(data)
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
            print("Connection test fail, please check the serial port connectivity.")
            #self.serial_port = None
            pos_target = [0, 0, 0]
            cur_target = [0, 0]
            pos = [0, 0, 0]
            cur = [0, 0]
            stf = [0, 0]
            pos_offset = [0, 0, 0]
            cur_offset = [0, 0]
            stf_offset = [0, 0]
            return False
        
        print("Connection confirm with return: %s" % (com))

        while self.pos is None:
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
        
        while self.cur is None:
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

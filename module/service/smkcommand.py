from enum import Enum

API_VERSION = "v7.2.0"
INFO_ALL = 0

BAUD_RATE_T_2000000    = 0           #//< Define to identify 2M baudrate
BAUD_RATE_T_460800     = 1           #//< Define to identify 460.8k baudrate
MAX_WATCHDOG_TIME      = 500         #//< Maximum watchdog timer time
READ_TIMEOUT           = 4000        #//< Timeout on readings

class smkcommandv1(Enum):
    #//=========================================================     general commands

    VERSION = 1
    BAUDRATE = 115200

    CMD_PING                    = 0    #< Asks for a ping message """
    CMD_START                   = 241
    CMD_STOP                    = 242
    CMD_START_ALL_CALIBRATE     = 244
    CMD_START_ONE_CALIBRATE     = 243   #need follow by channel number
    CMD_STOP_CALIBRATE          = 245


    #//=========================================================     qbcommands 

    CMD_ACTIVATE                = 128  #///< Command for activating/deactivating


    CHANNEL_NUM = 12

class smkcommandv2(Enum):
    #//=========================================================     general commands
    VERSION = 2
    BAUDRATE = 460800

    CMD_SETTING                    = "AT+EMGCONFIG=FFFFFFFF,2000\r\n"    #< Asks for a ping message """
    CMD_CHECK_SETTING             = "AT+EMGCONFIG?\r\n"
    CMD_START_IEMG             = "AT+IEMGSTRT\r\n"
    CMD_CHECK_IEMG              = 0x71
    CMD_START_EMG               = "AT+EMGSTRT\r\n"
    CMD_CHECK_EMG              = 0x74
    CMD_TRIGGER     = "AT+TRIGER=1\r\n"
    CMD_STOP     = "AT+STOP\r\n"   #need follow by channel number
    CMD_CHECK_VERSION          = "AT+SWVER?\r\n"

    CMD_START = CMD_START_IEMG

    CHANNEL_NUM = 32


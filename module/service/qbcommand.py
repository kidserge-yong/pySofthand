from enum import Enum

API_VERSION = "v6.2.0"
INFO_ALL = 0
BAUD_RATE = 2000000
BAUD_RATE_T_2000000    = 0           #//< Define to identify 2M baudrate
BAUD_RATE_T_460800     = 1           #//< Define to identify 460.8k baudrate
MAX_WATCHDOG_TIME      = 500         #//< Maximum watchdog timer time
READ_TIMEOUT           = 4000        #//< Timeout on readings

class qbmove_command(Enum):
    #//=========================================================     general commands

    CMD_PING                    = 0    #< Asks for a ping message """
    CMD_SET_ZEROS               = 1    #///< Command for setting the encoders zero position
    CMD_STORE_PARAMS            = 3    #///< Stores all parameters in memory and
                                        #///  loads them
    CMD_STORE_DEFAULT_PARAMS    = 4    #///< Store current parameters as factory parameters
    CMD_RESTORE_PARAMS          = 5    #///< Restore default factory parameters
    CMD_GET_INFO                = 6    #///< Asks for a string of information about

    CMD_SET_VALUE               = 7    #///< Not Used
    CMD_GET_VALUE               = 8    #///< Not Used

    CMD_BOOTLOADER              = 9    #///< Sets the bootloader modality to update the
                                        #///  firmware
    CMD_INIT_MEM                = 10   #///< Initialize the memory with the defalut values
    CMD_CALIBRATE               = 11   #///< Starts the stiffness calibration of the qbMove
    CMD_GET_PARAM_LIST          = 12   #///< Command to get the parameters list or to set
                                        #///  a defined value chosen by the use
    CMD_HAND_CALIBRATE          = 13   #///< Starts a series of opening and closures of the hand

    #//=========================================================     qbcommands 

    CMD_ACTIVATE                = 128  #///< Command for activating/deactivating
                                        #///  the device
    CMD_GET_ACTIVATE            = 129  #///< Command for getting device activation
                                        #///  state
    CMD_SET_INPUTS              = 130  #///< Command for setting reference inputs
    CMD_GET_INPUTS              = 131  #///< Command for getting reference inputs
    CMD_GET_MEASUREMENTS        = 132  #///< Command for asking device's
                                        #///  position measurements
    CMD_GET_CURRENTS            = 133  #///< Command for asking device's
                                        #///  current measurements
    CMD_GET_CURR_AND_MEAS       = 134  #///< Command for asking device's
                                        #///  measurements and currents
    CMD_SET_POS_STIFF           = 135  #///< Not used in the softhand firmware
    CMD_GET_EMG                 = 136  #///< Command for asking device's emg sensors 
                                        #///  measurements
    CMD_GET_VELOCITIES          = 137  #///< Command for asking device's
                                        #///  velocity measurements
    CMD_GET_COUNTERS            = 138  #///< Command for asking device's counters
                                        #///  (mostly used for debugging sent commands)
    CMD_GET_ACCEL               = 139  #///< Command for asking device's
                                        #///  acceleration measurements
    CMD_GET_CURR_DIFF           = 140  #///< Command for asking device's 
                                        #///  current difference between a measured
                                        #///  one and an estimated one (Only for SoftHand)
    CMD_SET_CURR_DIFF           = 141  #///< Command used to set current difference modality
                                        #///  (Only for Cuff device)
    CMD_SET_CUFF_INPUTS         = 142  #///< Command used to set Cuff device inputs 
                                        #///  (Only for Cuff device)
    CMD_SET_WATCHDOG            = 143  #///< Command for setting watchdog timer
                                        #///  or disable it
    CMD_SET_BAUDRATE            = 144  #///< Command for setting baudrate
                                        #///  of communication
    CMD_EXT_DRIVE               = 145  #///< Command to set the actual measurements as inputs
                                        #///  to another device (Only for Armslider device)
    CMD_GET_JOYSTICK            = 146   #///< Command to get the joystick measurements (Only 
                                        #///  for devices driven by a joystick)

class qbmove_parameter(Enum):
    PARAM_ID                     = 0   #///< Device's ID number
    PARAM_PID_CONTROL            = 1   #///< PID parameters
    PARAM_STARTUP_ACTIVATION     = 2   #///< Start up activation byte
    PARAM_INPUT_MODE             = 3   #///< Input mode

    PARAM_CONTROL_MODE           = 4   #///< Choose the kind of control between
                                        #///  position control current control
                                        #///  direct PWM value or current+position control
    PARAM_MEASUREMENT_OFFSET     = 5   #///< Adds a constant offset to the
                                        #///  measurements
    PARAM_MEASUREMENT_MULTIPLIER = 6   #///< Adds a multiplier to the
                                        #///  measurements
    PARAM_POS_LIMIT_FLAG         = 7   #///< Enable/disable position limiting
    PARAM_POS_LIMIT              = 8   #///< Position limit values
                                        #///  | int32     | int32     | int32     | int32     |
                                        #///  | INF_LIM_1 | SUP_LIM_1 | INF_LIM_2 | SUP_LIM_2 |
    PARAM_MAX_STEP_POS           = 9   #///< Used to slow down movements for positive values
    PARAM_MAX_STEP_NEG           = 10  #///< Used to slow down movements for negative values
    PARAM_POS_RESOLUTION         = 11  #///< Angle resolution for inputs and
                                        #///  measurements. Used during
                                        #///  communication.
    PARAM_CURRENT_LIMIT          = 12  #///< Limit for absorbed current
    PARAM_EMG_CALIB_FLAG         = 13  #///< Enable calibration on startup
    PARAM_EMG_THRESHOLD          = 14  #///< Minimum value to have effect
    PARAM_EMG_MAX_VALUE          = 15  #///< Maximum value of EMG
    PARAM_EMG_SPEED              = 16  #///< Closure speed when using EMG
    PARAM_PID_CURR_CONTROL       = 18  #///< PID current control
    PARAM_DOUBLE_ENC_ON_OFF      = 19  #///< Double Encoder Y/N
    PARAM_MOT_HANDLE_RATIO       = 20  #///< Multiplier between handle and motor
    PARAM_MOTOR_SUPPLY           = 21  #///< Motor supply voltage of the hand
    PARAM_CURRENT_LOOKUP         = 23  #///< Table of values used to calculate 
                                        #///  an estimated current of the SoftHand
    PARAM_DL_POS_PID             = 24  #///< Double loop position PID
    PARAM_DL_CURR_PID            = 25   #///< Double loop current PID

class qbmove_resolution(Enum):
    RESOLUTION_360      = 0
    RESOLUTION_720      = 1
    RESOLUTION_1440     = 2
    RESOLUTION_2880     = 3
    RESOLUTION_5760     = 4
    RESOLUTION_11520    = 5
    RESOLUTION_23040    = 6
    RESOLUTION_46080    = 7
    RESOLUTION_92160    = 8

class qbmove_input_mode(Enum):
    CONTROL_ANGLE           = 0        #///< Classic position control
    CONTROL_PWM             = 1        #///< Direct PWM value
    CONTROL_CURRENT         = 2        #///< Current control
    CURR_AND_POS_CONTROL    = 3        #///< Position and current control
    DEFLECTION_CONTROL      = 4        #///< Deflection control
    DEFL_CURRENT_CONTROL    = 5         #///< Deflection and current control   

class motor_supply_tipe(Enum):
    MAXON_24V               = 0
    MAXON_12V               = 1

class acknowledgment_values(Enum):
    ACK_ERROR           = 0
    ACK_OK              = 1

class data_types(Enum):
    TYPE_FLAG    = 0       #//A uint8 but with a menu
    TYPE_INT8    = 1
    TYPE_UINT8   = 2
    TYPE_INT16   = 3
    TYPE_UINT16  = 4
    TYPE_INT32   = 5
    TYPE_UINT32  = 6
    TYPE_FLOAT   = 7
    TYPE_DOUBLE  = 8

PARAM_BYTE_SLOT = 50
PARAM_MENU_SLOT = 150

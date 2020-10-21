class Qbrobot():

    ROBOT_NAME = "Qbrobot"

    @staticmethod
    def qbrobot_init(data):
        data[Qbrobot.ROBOT_NAME] = {}
        data[Qbrobot.ROBOT_NAME]["timeout"] = 0
        data[Qbrobot.ROBOT_NAME]["is_model_loaded"] = False
        data[Qbrobot.ROBOT_NAME]["jobs"] = []
        data[Qbrobot.ROBOT_NAME]["results"] = {}

class robot2(threading.Thread, Qbrobot):
    TIME_CYCLE = 0.01
    TIME_OUT = 30 * 30 # 900 second for 15 minutes without interaction

    part = []
    
    def __init__(self, global_data):
        threading.Thread.__init__(self)
        self.global_data = global_data

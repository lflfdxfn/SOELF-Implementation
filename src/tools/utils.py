import os
import multiprocessing
import traceback
import subprocess
import time
import platform

from subprocess import PIPE
from types import FunctionType
from src.tools.weakly_settings import weakly_settings

def get_origin_name(run_data, origin_data_list):
    for origin_data in origin_data_list:
        if origin_data in run_data:
            return origin_data

    return run_data

def long_path_transfer(input_path):
    if "Windows" not in platform.architecture()[1]:
        return input_path

    if "\\\\?\\" not in input_path:
        output_path = "\\\\?\\" + os.path.abspath(input_path)
        return output_path
    else:
        return input_path


def check_file_directory(method:FunctionType, run_data:str, weakly:weakly_settings, settings:dict):
    pred_path = settings["pred_path"]

    result_dir = long_path_transfer("{}/{}/{}/{}_{}_{:.2f}".format(pred_path, method.__name__, run_data, weakly.init_index, weakly.m, weakly.p))

    # whether run results dirs exists
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    return result_dir

def error(msg, *args):
    return multiprocessing.get_logger().error(msg, *args)

class LogExceptions(object):
    def __init__(self, callable):
        self.__callable = callable
        return

    def __call__(self, *args, **kwargs):
        try:
            result = self.__callable(*args, **kwargs)

        except Exception as e:
            # Here we add some debugging help. If multiprocessing's
            # debugging is on, it will arrange to log the traceback
            error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can
            # clean up
            raise

        # It was fine, give a normal answer
        return result

    pass

def throw_exception(name):
    cmd = 'taskkill /im ' + str(os.getpid()) + ' /F'
    res = subprocess.Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE)
    print(res.stdout.read())
    time.sleep(1)


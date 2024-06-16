import numpy as np
import pickle
import os

from types import FunctionType

import pandas as pd
from scipy.io import loadmat
from tools.weakly_settings import weakly_settings
from tools.utils import check_file_directory, LogExceptions, throw_exception
from src.tools.eval_sliding import eval_sliding_file, plot_sliding_results, wrst, plot_updation_result
from src.tools.utils import get_origin_name
from src.tools.utils import long_path_transfer


def middle_data_fetcher(method:FunctionType, run_data:str, weakly:weakly_settings, settings:dict, analyze_updation = True):
    pred_path = settings["pred_path"]
    init_index = weakly.init_index
    m = weakly.m
    p = weakly.p

    middle_dir = long_path_transfer("{}/{}/{}/".format(settings["middle_path"], method.__name__, run_data))
    if not os.path.exists(middle_dir):
        os.makedirs(middle_dir)
    middle_output_file = "{}\\{}_{}_{:.2f}.pyd".format(middle_dir, init_index, m, p)
    with open(middle_output_file, "rb") as file:
        middle_data = pickle.load(file)

    return middle_data
import numpy as np
import random
import multiprocessing
import pickle
import os

from types import FunctionType

import pandas as pd
from scipy.io import loadmat
from tools.weakly_settings import weakly_settings
from tools.utils import check_file_directory, LogExceptions, throw_exception
from algorithms.data_params import data_params
from src.tools.eval_sliding import eval_sliding_file, plot_sliding_results, wrst, plot_updation_result
from src.tools.utils import get_origin_name
from src.tools.utils import long_path_transfer

def single_run(method, result_dir, param, y_mask, i_run):
    print("Running... Serial number: {}".format(i_run))
    random.seed(i_run)
    np.random.seed(i_run)

    method(result_dir, param, y_mask, i_run)

def run_algorithm(method:FunctionType, run_data:str, weakly:weakly_settings, settings:dict, pool = None):

    # get current needed settings
    n_runs = settings["n_runs"]
    if_train_model = settings["if_train_model"]
    if_parallel = settings["if_parallel"]

    # fetch data and hyper parameters
    param = data_params(run_data, settings)
    # fetch weakly scenario data
    y_mask = weakly.weakly_mask(run_data)

    # print info
    weakly.print_info()
    print("Experimental Settings:\n\tNum of runs: {}".format(settings["n_runs"]))
    print("Method: {}".format(method.__name__))
    param.print_info()

    # prepare and check output directory
    result_dir = check_file_directory(method, run_data,weakly, settings)

    if if_train_model:
        if pool:
            # submit mission
            for i_run in range(n_runs):
                pool.apply_async(LogExceptions(single_run), args=(method, result_dir, param, y_mask, i_run), error_callback=throw_exception)
            # close pool
            pool.close()
            # wait each subprocess
            pool.join()
            # all works finished
        else:
            for i_run in range(n_runs):
                single_run(method, result_dir, param, y_mask, i_run)

def analyze_algorithm(method:FunctionType, run_data:str, weakly:weakly_settings, settings:dict, plot_results = True, analyze_updation = True):
    pred_path = settings["pred_path"]
    init_index = weakly.init_index
    m = weakly.m
    p = weakly.p

    pred_infos = {
        "pred_dir": "{}/{}/{}/{}_{}_{:.2f}".format(pred_path, method.__name__, run_data, init_index, m, p),
        "prefix": "run_",
        "suffix": ".txt",
        "init_index": init_index,
        "window_size": settings["window_sizes"],
        "n_runs": settings["n_runs"],
        "index_diff": -1
    }

    if "matlab" in method.__name__:
        pred_infos["pred_dir"] = "{}/{}/{}/{}_{}_{:.2f}/runs".format(pred_path, method.__name__, run_data, init_index, m, p)
        pred_infos["index_diff"] = 0

    ### store middle results
    sliding_recalls_runs, sliding_gmeans_runs = eval_sliding_file(**pred_infos)
    sliding_gmean_runs = sliding_gmeans_runs.mean(axis=0)
    middle_data = {
        "sliding_recalls_runs": sliding_recalls_runs,
        "sliding_gmean_runs": sliding_gmean_runs
    }
    middle_dir = long_path_transfer("{}/{}/{}/".format(settings["middle_path"], method.__name__, run_data))
    if not os.path.exists(middle_dir):
        os.makedirs(middle_dir)
    middle_output_file = os.path.join(middle_dir, "{}_{}_{:.2f}.pyd".format(init_index, m, p))
    with open(middle_output_file, "wb") as file:
        pickle.dump(middle_data, file)

    # store plot results of gmean and recalls
    if plot_results:
        plot_output = long_path_transfer("{}/{}/{}/{}_{}_{:.2f}".format(settings["plot_output_path"], method.__name__, run_data, init_index, m, p))
        if not os.path.exists(plot_output):
            os.makedirs(plot_output)
        plot_sliding_results(plot_output, run_data, method.__name__, sliding_recalls_runs, sliding_gmeans_runs, init_index, m, p)

    if analyze_updation:
        # store plot results of updation
        updation_infos = {
            "data": loadmat("{}/{}.mat".format(settings["data_path"], run_data)),
            "pred_dir": long_path_transfer("{}/{}/{}/{}_{}_{:.2f}".format(settings["pred_path"], method.__name__, run_data, init_index, m, p)),
            "output_dir": long_path_transfer("{}/{}/{}/{}_{}_{:.2f}".format(settings["plot_output_path"], method.__name__, run_data, init_index, m, p)),
            "prefix": "run_",
            "suffix": "_updation.pyd",
            "chunk_size": init_index,
            "n_runs": settings["n_runs"],
            "index_diff": -1
        }
        plot_updation_result(**updation_infos)

    return sliding_gmean_runs

def compare_algorithms(method:FunctionType, run_data:str, weakly:weakly_settings, settings:dict):
    # get parameters
    origin_name = get_origin_name(run_data, settings["origin_data"])
    middle_path = settings["middle_path"]
    init_index = weakly.init_index
    m = weakly.m
    p = weakly.p

    # table path
    output_table_file = long_path_transfer(os.path.join(settings["comp_output_path"], "{}.csv".format(method.__name__)))
    # initialize a table if no table is found
    if os.path.exists(output_table_file):
        # existing table
        table = pd.read_csv(output_table_file, index_col = 0)

        for data_name_ in settings["origin_data"] + settings["real_data"]:
            # add another datasets
            if data_name_ not in table["dataset"].values:
                for m_ in settings["weakly_m"]:
                    for p_ in settings["weakly_p"]:
                        table.loc[len(table.index)] = [data_name_, m_, p_] + ["-"] * (len(table.columns) - 3)

            # add another scenarios
            for m_ in settings["weakly_m"]:
                for p_ in settings["weakly_p"]:
                    row_loc = (table["dataset"] == data_name_) & (table["m"] == m_) & (table["p"] == p_)
                    if sum(row_loc) == 0:
                        table.loc[len(table.index)] = [data_name_, m_, p_] + ["-"] * (len(table.columns) - 3)

    else:
        # blank table
        table = pd.DataFrame(columns=["dataset", "m", "p"])
        for data_name_ in settings["origin_data"]+settings["real_data"]:
            for m_ in settings["weakly_m"]:
                for p_ in settings["weakly_p"]:
                    table.loc[len(table.index)] = [data_name_, m_, p_]
        table[method.__name__] = "-"
        for comp_method in settings["compare_methods"]:
            table[comp_method] = "-"

    # results of the method on run_data
    method_middle_output_file = long_path_transfer("{}/{}/{}/{}_{}_{:.2f}.pyd".format(settings["middle_path"], method.__name__, run_data, init_index, m, p))
    with open(method_middle_output_file, 'rb') as file:
        method_middle_output = pickle.load(file)
    method_sliding_gmean_runs = method_middle_output["sliding_gmean_runs"]
    result_string = wrst(method_sliding_gmean_runs)
    row_loc = (table["dataset"] == origin_name) & (table["m"] == m) & (table["p"] == p)
    table.loc[row_loc, method.__name__] = result_string

    # compared results
    for comp_method in settings["compare_methods"]:
        comp_middle_output_file = long_path_transfer("{}/{}/{}/{}_{}_{:.2f}.pyd".format(settings["middle_path"], comp_method, run_data, init_index, m, p))
        with open(comp_middle_output_file, 'rb') as file:
            comp_middle_output = pickle.load(file)
        comp_sliding_gmean_runs = comp_middle_output["sliding_gmean_runs"]
        result_string = wrst(comp_sliding_gmean_runs, method_sliding_gmean_runs)

        row_loc = (table["dataset"] == origin_name) & (table["m"] == m) & (table["p"] == p)
        table.loc[row_loc, comp_method] = result_string

    table.to_csv(output_table_file)
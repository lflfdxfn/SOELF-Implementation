import multiprocessing

from algorithms.SOELF import SOELF
from run import run_algorithm, analyze_algorithm, compare_algorithms
from tools.weakly_settings import weakly_settings
from tools.eval_sliding import eval_sliding_file, wrst

# experimental settings
settings = {
    # methods chosen
    "exp_methods" : [SOELF],
    "compare_methods": [],

    # origin_dataset info
    "origin_data" : ["letter", "statlog", "covertype", "mnist"],
    "real_data": ["laden_ce", "wed_ce", "christ_ce","kddcup99", "pokerlsn", "huge_tweet"],
    "data_path" : "../DataStream/",
    "mask_path" : "../DataStream/label_mask",
    "middle_path": "../Analyze/middle_runs_result/",
    "plot_output_path" : "../Analyze/plot_analyze",
    "comp_output_path" : "../results",
    "pred_path" : "../predictions",
    "scenarios" : [1],
    "n_cases" : [1],

    # weakly scenario info
    "window_sizes" : 200,
    "init_indexes" : [1000],
    "weakly_m" : [10, 100],
    "weakly_p" : [0, 0.05, 0.10],

    # running info
    "if_train_model" : True,
    "if_parallel" : 10,
    "n_runs" : 10
}

# run data names
data_names =  ["{}_Scenario{}_Case{}".format(data, scenario, case+1)
              for data in settings["origin_data"]
              for scenario, n_case in zip(settings["scenarios"], settings["n_cases"])
              for case in range(n_case)] + settings["real_data"]

# weakly scenarios
weakly_scenarios = [{"init_index": init_index, "m": m, "p": p, "mask_path": settings["mask_path"]}
                    for init_index in settings["init_indexes"]
                    for m in settings["weakly_m"]
                    for p in settings["weakly_p"]]

if __name__=="__main__":
    # loop running
    # run datasets one-by-one
    for run_data in data_names:
        for method in settings["exp_methods"]:
            for weakly_scenario in weakly_scenarios:
                weakly = weakly_settings(weakly_scenario)

                pool = None
                if (settings["if_parallel"]>0) and (settings["if_train_model"]==1):
                    # create process pool
                    multiprocessing.log_to_stderr()
                    pool = multiprocessing.Pool(processes=settings["if_parallel"])

                run_algorithm(method, run_data, weakly, settings, pool)
                sliding_gmean_runs = analyze_algorithm(method, run_data, weakly, settings, plot_results=False, analyze_updation=False)
                compare_algorithms(method, run_data, weakly, settings)
                print()


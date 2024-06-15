import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import random
import warnings
import os

from scipy.stats import norm

plt.rcParams['figure.figsize'] = [12, 7]
plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'lines.linewidth': 4})

# Data Loading
pure_dataset_path = "./offline_datasets"
letter = pd.read_csv(os.path.join(pure_dataset_path, "letter.csv"))
statlog = pd.read_csv(os.path.join(pure_dataset_path, "statlog.csv"))
covertype = pd.read_csv(os.path.join(pure_dataset_path, "covertype.csv"))
mnist = pd.read_csv(os.path.join(pure_dataset_path, "mnist.csv"))
datasets = {"letter":letter,
            "statlog":statlog,
            "mnist":mnist,
            "covertype":covertype}

# Methods
def standardize_01(array):
    return lambda x: x / (max(array))

def standardize_maxvalue(array, max_value):
    return lambda x: max_value * x / (max(array))

def stream_plot(_plot_curves, _chunk_size, _n_chunk, _title, _chunk_text = False):

    chosen_linestyle = [
        "solid",
        "dashed",
        "dashdot",
        (0, (3, 1, 1, 1)),  # densely dashdotted
        "dotted",
        (0, (3, 5, 1, 5)), # dashdotted
        (0, (3, 5, 1, 5, 1, 5))# dashdotdotted
    ]

    # sort by keys
    _plot_curves = dict(sorted(_plot_curves.items()))

    # key paramters
    _length = _chunk_size*_n_chunk
    _t = range(1, _length + 1)

    ### plot prior prob
    _plot_figure = plt.figure()
    # _plot_figure, _axes = plt.subplots(1, 1)
    for idx_curve, key in enumerate(_plot_curves.keys()):
        plt.plot(_t, _plot_curves[key] + 0.002 * (-1) ** idx_curve, label="class {}".format(idx_curve + 1), linestyle = chosen_linestyle[idx_curve%len(chosen_linestyle)])

    plt.title(_title)
    plt.legend(loc='lower right')
    plt.grid()
    plt.ylim([0, 1])
    plt.xlim([0, _length])
    plt.ylabel("Prior Probability")
    plt.xlabel(r"Time Step/t")

    return _plot_figure

def stream_generation(_data_table, _curves, _chunk_size, _n_chunk):
    # key parameters
    _length = _chunk_size * _n_chunk
    base_X = _data_table.iloc[:, :-1].values
    base_Y = _data_table.iloc[:, -1].values
    labels, _ = np.unique(base_Y, return_counts=True)
    label_dataX = dict()
    for label in labels:
        label_dataX[label] = base_X[base_Y == label]

    # begin to generate data streams!
    # sort by keys
    _curves = dict(sorted(_curves.items()))
    # prob matrix
    data_list = [_curves[class_] for class_ in _curves]
    prob_matrix = np.column_stack(data_list)
    # precision
    threshold = 1e-10
    prob_matrix[prob_matrix<=threshold] = 0

    # data stream
    x_stream = np.empty((_length, base_X.shape[1]))
    x_stream[:] = np.NaN
    y_stream_label = np.array([-1] * _length)
    y_stream = np.empty((_length, 1), dtype=int)
    y_stream[:] = -1

    for idx_ins, prob in enumerate(prob_matrix):
        # ALERT!!!!!! the class label must begin from 1 not 0
        y_stream[idx_ins, 0] = np.random.choice(len(_curves.keys()), p = prob) + 1
        y_stream_label[idx_ins] = list(_curves.keys())[y_stream[idx_ins, 0] - 1]

    labels, counts = np.unique(y_stream_label, return_counts=True)

    for label, count in zip(labels, counts):
        base_label_size = label_dataX[label].shape[0]
        if base_label_size <= count:
            select_index = np.random.choice(base_label_size, replace=True, size=count)
        else:
            select_index = np.random.choice(base_label_size, replace=False, size=count)

        x_stream[np.isin(y_stream_label, label), :] = label_dataX[label][select_index, :]

    x_stream_trans = x_stream.transpose()

    return x_stream_trans, y_stream, prob_matrix

def disp_scenarios(_data_name, _folder_name, _n_scenario, _n_case, _n_existing, _disp_point, _zero_dura, _reoccur_dura, _gaussian_std, _chunk_size, _n_chunk = 15, _seed = 42):
    if_success = False

    # key settings and parameters
    random.seed(_seed)
    np.random.seed(_seed)
    data_table = datasets[_data_name]
    length = int(_chunk_size * _n_chunk)
    max_classes = len(data_table['class'].value_counts())

    if _n_existing <= max_classes:
        # existing classes
        existing_classes = list(range(1, _n_existing + 1))
        # disappearing class
        disp_class = existing_classes[-1]

        # initialize for curves generation
        t = np.array(range(length))
        curves = dict()

        # annotation of exisiting lower bound or not
        lower_bounds = dict()

        # gen two gaussian curve
        # the first, disappering curve
        mean_t1 = int(_chunk_size * _disp_point)
        std1 = _gaussian_std * _chunk_size
        gen_curve1 = norm.pdf(t, mean_t1, std1)
        gen_curve1 = standardize_maxvalue(gen_curve1, 1/_n_existing)(gen_curve1)
        # the second, reoccurring curve
        reoccur_point = _disp_point + 2 * int(_gaussian_std * 3) + _zero_dura
        mean_t2 = int(_chunk_size * reoccur_point)
        std2 = _gaussian_std * _chunk_size
        gen_curve2 = norm.pdf(t, mean_t2, std2)
        gen_curve2 = standardize_maxvalue(gen_curve2, 1/_n_existing)(gen_curve2)

        # integrate
        curves[disp_class] = np.concatenate([np.array([1/_n_existing] * mean_t1),
                                             gen_curve1[mean_t1:(mean_t1+ _chunk_size * int(_gaussian_std*3))],
                                             np.array([0] * int(_zero_dura*_chunk_size)),
                                             gen_curve2[(mean_t2-_chunk_size*int(_gaussian_std*3)):mean_t2],
                                             np.array([1/_n_existing] * int(_reoccur_dura * _chunk_size))])

        # curves of existing(all the time) classes
        for e_class in existing_classes[:-1]:
            curves[e_class] = np.array([0.0]*length)
            # generate corresponding prior probability
            curves[e_class] = (1 - curves[disp_class]) / (len(existing_classes)-1)


        # plot prob curves
        save_header = "{}_Scenario{}_Case{}".format(_data_name, _n_scenario, _n_case)
        fig = stream_plot(curves, _chunk_size, _n_chunk, save_header)
        fig.show()
        # generate data
        x_stream, y_stream, prob_matrix = stream_generation(data_table, curves, _chunk_size, _n_chunk)

        # save figure
        fig.savefig("{}/{}.jpg".format(_folder_name, save_header))

        # plot real data distribution
        label_plot_x = np.array(range(1, len(y_stream) + 1))
        label_plot_y = np.squeeze(y_stream)
        plt.figure()
        plt.title(save_header)
        plt.scatter(label_plot_x, label_plot_y, s = 6)
        plt.xlabel("Time Step/t")
        plt.ylabel("Class Label")

        plt.yticks(np.arange(1, _n_existing + 1, step = 1))
        plt.grid()
        plt.savefig("./{}/{}_data.jpg".format(_folder_name, save_header))

        sio.savemat("./{}/{}.mat".format(_folder_name, save_header),
                    {'data_name': _data_name,
                     'folder_name': _folder_name,
                     'n_scenario': _n_scenario,
                     'n_case': _n_case,
                     'n_existing': _n_existing,
                     'disp_point': _disp_point,
                     'zero_dura': _zero_dura,
                     'reoccur_dura': _reoccur_dura,
                     'gaussian_std': _gaussian_std,
                     'chunk_size': _chunk_size,
                     'n_chunk': _n_chunk,
                     'seed': _seed,
                     'save_header': save_header,
                     'x': x_stream, 'y': y_stream, 'prior': prob_matrix})

        if_success = True

    else:
        warnings.warn("The original dataset does not have enough classes for generation!")

    return if_success

if __name__ == "__main__":
    n_scenario = 2
    n_existing = 3
    disp_point = 2
    zero_dura = 3
    reoccur_dura = 2
    chunk_size = 500
    seed = 0
    gaussian_std = 1/3

    folder_name = "./"

    for data_name in ["covertype", "letter", "statlog", "mnist"]:
        n_case = 1

        n_chunk = disp_point + 2*int(gaussian_std*3) + zero_dura + reoccur_dura

        status = disp_scenarios(_data_name=data_name,
                               _folder_name=folder_name,
                               _n_scenario=n_scenario,
                               _n_case=n_case,
                               _n_existing=n_existing,
                               _disp_point=disp_point,
                               _zero_dura=zero_dura,
                               _reoccur_dura=reoccur_dura,
                               _gaussian_std=gaussian_std,
                               _chunk_size=chunk_size,
                               _n_chunk=n_chunk,
                               _seed=seed)

        if status:
            print("Scenario {} Cases {} Completed!".format(n_scenario, n_case))
        n_case = n_case + 1

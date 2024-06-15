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
offline_dataset_path = "./offline_datasets"
letter = pd.read_csv(os.path.join(offline_dataset_path, "letter.csv"))
statlog = pd.read_csv(os.path.join(offline_dataset_path, "statlog.csv"))
covertype = pd.read_csv(os.path.join(offline_dataset_path, "covertype.csv"))
mnist = pd.read_csv(os.path.join(offline_dataset_path, "mnist.csv"))
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

def stream_generation(_data_table, _curves, _chunk_size, _n_chunk, _emer_threshold):
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
    threshold = _emer_threshold
    prob_matrix[prob_matrix<=threshold] = 0
    prob_matrix = prob_matrix/np.expand_dims(prob_matrix.sum(axis= 1), axis=1)

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

def scenarios(_data_name, _folder_name, _n_scenario, _n_case, _n_existing, _n_emerging, _max_values, _mean_points, _gaussian_stds, _emer_threshold, _disp_or_nots, _e_durations, _chunk_size = 200, _n_chunk = 15, _seed = 42):
    if_success = False

    # key settings and parameters
    random.seed(_seed)
    np.random.seed(_seed)
    data_table = datasets[_data_name]
    length = _chunk_size * _n_chunk
    max_classes = len(data_table['class'].value_counts())

    if _n_existing + _n_emerging <= max_classes:
        # existing classes
        existing_classes = list(range(1, _n_existing + 1))
        emerging_classes = list(range(_n_existing + 1, _n_existing + _n_emerging + 1))

        # initialize for curves generation
        t = np.array(range(length))
        curves = dict()

        # curves of emerging classes
        emer_prob = np.zeros(length)
        # annotation of exisiting lower bound or not
        lower_bounds = dict()
        for e_class, max_value, mean_point, gaussian_std, disp_or_not, e_dura in zip(emerging_classes, _max_values, _mean_points, _gaussian_stds, _disp_or_nots, _e_durations):
            # gen a gaussian curve
            mean_t = int(_chunk_size * mean_point)
            std = gaussian_std * _chunk_size
            len_duration = _chunk_size * e_dura
            # assign duration variation
            gen_curve = norm.pdf(t, mean_t, std)
            gen_curve = standardize_maxvalue(gen_curve, max_value)(gen_curve)

            curves[e_class] = np.concatenate([gen_curve[:mean_t], np.array([max(gen_curve)] * len_duration), gen_curve[mean_t:]], axis=0)[:length]
            # accumulate the sum of prior probability
            emer_prob += curves[e_class]

            # use lower bounds for classes that keep existing after emerging
            if disp_or_not:
                # disappear
                lower_bounds[e_class] = [0] * length
            else:
                # existing after emerging
                lower_bounds[e_class] = [0] * (mean_t + len_duration) + [1] * length
                lower_bounds[e_class] = lower_bounds[e_class][:length]
        # standardization for emerging classes
        emer_disp_transform = standardize_maxvalue(emer_prob, max(_max_values))
        emer_prob = emer_disp_transform(emer_prob)
        for e_class in emerging_classes:
            curves[e_class] = emer_disp_transform(curves[e_class])

        # curves of existing classes
        for e_class in existing_classes:
            curves[e_class] = np.array([0.0]*length)
        # generate corresponding prior probability
        for idx_t in range(length):
            # rest prob for existing classes
            lower_prob = (1 - emer_prob[idx_t]) / len(existing_classes)

            # find the emerging and then existing classes for makeup
            lower_emer_classes = []
            for e_class in emerging_classes:
                if (lower_bounds[e_class][idx_t] == 1) and (curves[e_class][idx_t] < lower_prob):
                    lower_emer_classes.append(e_class)
            fixed_emer_classes = np.array(emerging_classes)[~np.isin(emerging_classes, lower_emer_classes)]

            # makeup for emerging and then existing classes
            rest_prob = 1
            for e_class in fixed_emer_classes:
                rest_prob -= curves[e_class][idx_t]

            makeup_prob = rest_prob / (len(existing_classes) + len(lower_emer_classes))
            for e_class in lower_emer_classes + existing_classes:
                curves[e_class][idx_t] = makeup_prob

        # plot prob curves
        save_header = "{}_Scenario{}_Case{}".format(_data_name, _n_scenario, _n_case)
        fig = stream_plot(curves, _chunk_size, _n_chunk, save_header)
        # generate data
        x_stream, y_stream, prob_matrix = stream_generation(data_table, curves, _chunk_size, _n_chunk, _emer_threshold)

        # save figure
        fig.savefig("./{}/{}.jpg".format(_folder_name, save_header))

        # plot real data distribution
        label_plot_x = np.array(range(1, len(y_stream) + 1))
        label_plot_y = np.squeeze(y_stream)
        plt.figure()
        plt.title(save_header)
        plt.scatter(label_plot_x, label_plot_y, s = 6)
        plt.xlabel("Time Step/t")
        plt.ylabel("Class Label")

        plt.yticks(np.arange(1, _n_existing + n_emerging + 1, step = 1))
        plt.grid()
        plt.savefig("./{}/{}_data.jpg".format(_folder_name, save_header))

        sio.savemat("./{}/{}.mat".format(_folder_name, save_header),
                    {'data_name': _data_name,
                     'folder_name': _folder_name,
                     'n_scenario': _n_scenario,
                     'n_case': _n_case,
                     'n_existing': _n_existing,
                     'n_emerging': _n_emerging,
                     'max_values': _max_values,
                     'mean_points': _mean_points,
                     'gaussian_stds': _gaussian_stds,
                     'disp_or_nots': _disp_or_nots,
                     'e_durations': _e_durations,
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

    # Class Emergence
    n_scenario = 1
    n_existing = 2
    n_emerging = 1
    mean_points = [3]
    max_values = [1/3]
    disp_or_nots = [1]
    e_durations = [3]
    n_chunk = 9
    chunk_size = 500
    seed = 0
    gaussian_stds = [1 / 3] * n_emerging
    emer_threshold = 0.005

    folder_name = "./"

    for data_name in ["letter", "statlog", "covertype", "mnist"]:
        n_case = 1

        status = scenarios(_data_name = data_name,
                           _folder_name = folder_name,
                           _n_scenario = n_scenario,
                           _n_case = n_case,
                           _n_existing = n_existing,
                           _n_emerging = n_emerging,
                           _max_values = max_values,
                           _mean_points = mean_points,
                           _gaussian_stds = gaussian_stds,
                           _emer_threshold = emer_threshold,
                           _disp_or_nots = disp_or_nots,
                           _e_durations = e_durations,
                           _chunk_size=chunk_size,
                           _n_chunk=n_chunk,
                           _seed=seed)

        if status:
            print("Scenario {} Cases {} Completed!".format(n_scenario, n_case))
            n_case = n_case + 1
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from scipy.stats import mannwhitneyu
from src.tools.utils import long_path_transfer

def eval_sliding_file(pred_dir, prefix = "", suffix = ".txt", init_index = 1000, window_size = 200, n_runs = 10, index_diff = 0, end_point = None):
    sliding_recalls_runs = []

    # loop over all runs
    for i_run in range(n_runs):
        pred_file = long_path_transfer(os.path.join(pred_dir, "{}{}{}".format(prefix, i_run+1 + index_diff, suffix)))
        pred_data = pd.read_csv(pred_file, delimiter=' ', header=None)
        true_label = pred_data.iloc[:, 0].values
        pred_label = pred_data.iloc[:, 1].values

        # initialize
        if i_run == 0:
            n_example = pred_data.shape[0]
            n_class = len(np.unique(true_label))
            n_eval = n_example - init_index

            sliding_recalls_runs = np.empty([n_eval - window_size + 1, n_class, n_runs])
            sliding_recalls_runs [:] = np.nan

        # loop over all evaluated predictions
        for i_eval in range(n_eval - window_size + 1):
            window_begin = init_index + i_eval
            window_end = window_begin + window_size

            window_true = true_label[window_begin:window_end]
            window_pred = pred_label[window_begin:window_end]
            correct_or_not = window_true==window_pred

            # find each class
            window_class = np.unique(window_true)
            for _class in window_class:
                _class_mask = window_true == _class

                sliding_recalls_runs[i_eval, _class-1, i_run] = sum(correct_or_not[_class_mask]) / sum(_class_mask)

    prod_without_nan = sliding_recalls_runs.prod(axis = 1, where = ~np.isnan(sliding_recalls_runs))
    num_without_nan = 1 / (~np.isnan(sliding_recalls_runs)).sum(axis = 1)
    sliding_gmeans_runs = prod_without_nan ** num_without_nan

    return sliding_recalls_runs, sliding_gmeans_runs

def plot_sliding_results(output_path, data_name, method_name, sliding_recalls_runs, sliding_gmeans_runs, init_indexes, m, p):
    # plot settings
    linewidth = 3
    fontsize = 15

    # title prefix
    title = "Data Stream: {}, Method: {}".format(data_name, method_name)
    subtitle = "Init: {}, m: {}, p: {:2f}".format(init_indexes, m, p)

    (n_valued, n_classes, n_runs) = sliding_recalls_runs.shape

    # G-mean part
    sliding_gmeans = sliding_gmeans_runs.mean(axis=1)
    plt.figure()
    plt.plot(sliding_gmeans, linewidth = linewidth)
    plt.xlabel("Data Stream (t)", fontsize = fontsize)
    plt.ylabel("G-mean", fontsize = fontsize)
    plt.grid()
    plt.suptitle(title)
    plt.title(subtitle)
    plt.savefig(long_path_transfer(os.path.join(output_path, "g-mean.jpg")))
    plt.close()

    # Recalls part
    sliding_recalls = sliding_recalls_runs.mean(axis=2)
    plt.figure()
    for i in range(n_classes):
        plt.plot(sliding_recalls[:, i], linewidth=linewidth, label = "class {}".format(i+1))

    plt.xlabel("Data Stream (t)", fontsize=fontsize)
    plt.ylabel("Recall", fontsize=fontsize)
    plt.grid()
    plt.legend()
    plt.suptitle(title)
    plt.title(subtitle)
    plt.savefig(long_path_transfer(os.path.join(output_path, "recalls.jpg")))
    plt.close()

    # Each Recall
    for i in range(n_classes):
        plt.figure()
        plt.plot(list(range(n_valued)), sliding_recalls[:, i], linewidth=linewidth, label="class {}".format(i + 1))
        plt.xlabel("Data Stream (t)", fontsize=fontsize)
        plt.ylabel("Recall", fontsize=fontsize)
        plt.xlim([0, n_valued-1])
        plt.grid()
        plt.legend()
        plt.suptitle(title)
        plt.title(subtitle)
        plt.savefig(long_path_transfer(os.path.join(output_path, "recall_class{}.jpg".format(i+1))))
        plt.close()

def plot_updation_result(data, pred_dir, output_dir, prefix, suffix, chunk_size =1000, n_runs = 10, index_diff = 0):
    # get data
    x = data["x"]
    y = data["y"]
    (n_dim, n_example) = x.shape
    n_class = len(np.unique(y))
    n_chunk = int(np.ceil(n_example/chunk_size))
    print("Analyze Settings:")
    print("\tChunk Size: {}".format(chunk_size))
    print("\tNum of Chunk: {}".format(n_chunk))

    # begin analyze
    for i_run in range(n_runs):
        # fetch data
        pred_file = os.path.join(pred_dir, "{}{}{}".format(prefix, i_run+1 + index_diff, suffix))
        with open(pred_file, 'rb') as file:
            updation_result = pickle.load(file)

        # prepare figure plot
        plt.figure(figsize = (10, 9))
        n_subplots = len(updation_result.keys())

        # prepare table to store
        # table output
        table = pd.DataFrame([], columns=["info"] + list(range(1, n_chunk+1)))

        # begin
        sorted_updation_result = dict(sorted(updation_result.items()))
        for i_subplot, (class_label, pn_updation) in enumerate(sorted_updation_result.items()):
            class_label = int(class_label)

            chunk_right_updation = np.zeros([n_chunk, n_class + 1])
            chunk_wrong_updation = np.zeros([n_chunk, 2])
            chunk_negative_updation = np.zeros([n_chunk, n_class + 1])

            p_updation = pn_updation["positive"]
            n_updation = pn_updation["negative"]

            for (t_count, real_label, update_time) in p_updation:
                idx_chunk = int(np.ceil((t_count+1)/chunk_size) - 1)

                if real_label == class_label:
                    chunk_right_updation[idx_chunk, 0] += update_time
                else:
                    chunk_wrong_updation[idx_chunk, 0] += update_time

            for (t_count, real_label, update_time) in n_updation:
                idx_chunk = int(np.ceil((t_count + 1) / chunk_size) - 1)

                chunk_negative_updation[idx_chunk, int(real_label)] += update_time

                if real_label != class_label:
                    chunk_right_updation[idx_chunk, int(real_label)] += update_time
                else:
                    chunk_wrong_updation[idx_chunk, 1] += update_time

            chunk_positive_updation = chunk_right_updation[:, 0] + chunk_wrong_updation[:, 0]

            np.seterr(divide='ignore', invalid='ignore')
            rate_right_positive_updation = chunk_right_updation[:, 0]/chunk_positive_updation
            rate_right_negative_updation = np.sum(chunk_right_updation[:, 1:], axis = 1) / np.sum(chunk_negative_updation, axis = 1)
            np.seterr(divide='warn', invalid='warn')


            # bar plot
            ax1 = plt.subplot(n_subplots, 1, i_subplot+1)
            plt.title("OVA of class {}".format(class_label))
            ax1.set_xlabel("Data Stream (Unit: Chunk)", labelpad = 0)
            ax1.set_ylabel("Counts")
            ax1.bar(list(range(1, n_chunk + 1)), chunk_positive_updation, label = "P class {}".format(class_label))
            bottom = chunk_positive_updation.copy()
            for negative_class in range(1, n_class+1):
                if negative_class != class_label:
                    ax1.bar(list(range(1, n_chunk + 1)), chunk_negative_updation[:, negative_class], label = "N class {}".format(negative_class), bottom = bottom)
                    bottom += chunk_negative_updation[:, negative_class]

            ax1.legend(loc= "upper left")

            # ratio plot
            ax2 = ax1.twinx()
            ax2.plot(list(range(1, n_chunk + 1)), rate_right_positive_updation, label = "P Right Ratio", color = "r", linewidth = 2, marker = "d", markersize = 5)
            ax2.plot(list(range(1, n_chunk + 1)), rate_right_negative_updation, label = "N Right Ratio", color = "purple", linewidth = 2, marker = "d", markersize = 5)
            ax2.legend(loc = "upper right")
            ax2.set_ylim([0,1])

            # table plot
            text_all = []
            text_positive = []
            text_negative = []
            for i_x in range(1, n_chunk+1):
                text_positive.append( "{:.2f}/({:.2f})".format(chunk_positive_updation[i_x-1], rate_right_positive_updation[i_x-1]))
            text_all.append(text_positive)
            for i_x in range(1, n_chunk+1):
                text_negative.append( "{:.2f}/({:.2f})".format(np.sum(chunk_negative_updation, axis = 1)[i_x-1], rate_right_negative_updation[i_x-1]))
            text_all.append(text_negative)
            # plt.xticks([])
            # plt.table(cellText=text_all, rowLabels= ["Positive", "Negative"], fontsize = 10, loc = "bottom", colLabels = [str(x_label) for x_label in range(1, n_chunk+1)])
            # plt.subplots_adjust(left = 0.2, bottom = 0.2)

            # store in table
            table.loc[len(table.index)] = ["OVA of class {} + ".format(class_label)] + text_positive
            table.loc[len(table.index)] = ["OVA of class {} - ".format(class_label)] + text_negative

            plt.grid()

        plt.tight_layout()
        plt.savefig(long_path_transfer("{}\\run_{}.jpg".format(output_dir, i_run)))
        table.to_csv(long_path_transfer("{}\\run_{}.csv".format(output_dir, i_run)), index = None)
        plt.close()

def wrst(comp_gmean_runs, base_gmean_runs = None, p_thre = 0.05):
    # mean/std values of compared gmean
    mean_comp_gmean = comp_gmean_runs.mean()
    std_comp_gmean = comp_gmean_runs.std(ddof = 1)

    suffix = ""

    # if have base method for comparison
    if base_gmean_runs is not None:
        # p-value calculate
        res = mannwhitneyu(comp_gmean_runs, base_gmean_runs) # non-paramatric wilcoxon rank-sum test
        p = res.pvalue

        # mean/std values of base gmean
        mean_base_gmean = base_gmean_runs.mean()
        std_base_gmean = base_gmean_runs.std(ddof = 1)

        if (mean_comp_gmean > mean_base_gmean) and (p<p_thre):
            suffix = " +"
        elif (mean_comp_gmean < mean_base_gmean) and (p<p_thre):
            suffix = " -"
        else:
            suffix = " ="

    return "{:.4f}/{:.4f}".format(mean_comp_gmean, std_comp_gmean) + suffix

if __name__=="__main__":
    sliding_recalls_runs, sliding_gmeans_runs = eval_sliding_file("test/covertype_100_0_CBCE_all", prefix ="run_", init_index = 0)
    sliding_recalls_runs2, sliding_gmeans_runs2 = eval_sliding_file("test/covertype_100_0_CBCE_all_initialize", prefix="run_", init_index=0)
    sliding_recalls_runs3, sliding_gmeans_runs3 = eval_sliding_file("test/covertype_100_0_EOEF_all_self_training", prefix="run_", init_index=0)

    gmean_runs = sliding_gmeans_runs.mean(axis = 0)
    gmean_runs2 = sliding_gmeans_runs2.mean(axis=0)
    gmean_runs3 = sliding_gmeans_runs3.mean(axis=0)

    print(wrst(gmean_runs2, gmean_runs)) #('0.3129/0.0240 +', 0.00018267179110955002)
    print(wrst(gmean_runs3, gmean_runs)) #('0.0752/0.0052 -', 0.00018267179110955002)
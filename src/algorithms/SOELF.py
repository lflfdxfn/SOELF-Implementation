import numpy as np
import os
import time
import subprocess
import pickle
import math
import random

from river import cluster
from river import stream
from scipy.stats import poisson
from scipy.io import savemat
from src.basic.ratio_update import classRatioUpdate
from src.algorithms import data_params
from src.basic.klr import OnlineKLR, Ensemble
from scipy.special import softmax
from src.basic.onlineclu import cluster_prediction, cluster_generate, auto_eps_denstream


def SOELF(result_dir: str, param: data_params, y_mask: np.array, i_run: int):
    # prepare hyper parameters
    M = 10
    # online klr parameters
    eta = param.eta
    lamda = param.lamda
    kernel_t = param.t
    cnt = 5000
    # online parameters
    ratio_decay = param.decay_factor
    disp_threshold = 1e-5
    # soelf parameters
    k_inverse = 3.5
    # parameters for DenStream:
    cluster_decay_factor = 0.01
    #

    # class info
    class_exist = np.array([])
    class_ratio = np.array([], dtype = float)
    class_ratio_initial = np.array([], dtype = float)

    # Models list for classes
    ensemble = []

    # Cluster models list for classes
    cluster_models = []
    #

    # Record recall of classifier and clustering model
    recall_classifier = []
    recall_cluster = []
    #

    # output and log
    file_result = os.path.join(result_dir, "run_{}.txt".format(i_run))
    detailed_result = os.path.join(result_dir, "run_{}_detailed.txt".format(i_run))
    time_result = os.path.join(result_dir, "run_{}.mat".format(i_run))
    fid_result = open(file_result, 'w')
    det_result = open(detailed_result, 'w')

    # fetch data
    x = param.x
    y = param.y
    num_classes = param.data_n_classes
    (dimension, example_count) = (param.data_n_dim, param.data_n_cnt)

    # updation info
    number_updation = {}
    updation_result = os.path.join(result_dir, "run_{}_updation.pyd".format(i_run))

    # run
    begin_time = time.time()
    for t_count in range(example_count):
        real_label = y[t_count]
        new_example = x[:, [t_count]]

        dict_x_i = {idx_d: value for idx_d, value in enumerate(np.squeeze(new_example))}

        # classify examples
        classifier_count = len(ensemble)
        ft_array = np.array([], dtype = float)
        if classifier_count > 0:
            classify_result = np.zeros(classifier_count)
            ft_array = np.zeros([M, classifier_count], dtype = float)

            for i in range(classifier_count):
                (classify_result_tmp, ft_array[:, i]) = ensemble[i].classify(new_example)

                classify_result[i] = sum(classify_result_tmp)/M

            max_probability = np.nanmax(classify_result)
            predic_subscript = np.take(np.argwhere(classify_result == max_probability), 0)
            prediction = class_exist[predic_subscript]
        else:
            prediction = 0
            max_probability = 0.5

        # store result
        fid_result.write("{} {} {:.8f}\n".format(int(real_label), int(prediction), max_probability))
        # store log
        detailed_str = "{} {} ".format(int(real_label), int(prediction))
        detailed_pred_probs = np.ones(num_classes) * (-1)
        for i_cls in range(len(ensemble)):
            label_cls = class_exist[i_cls]
            predprob_cls = classify_result[i_cls]
            detailed_pred_probs[int(label_cls) - 1] = predprob_cls

        preprobs_str = " ".join(map(lambda x: "{:.8f}".format(x), detailed_pred_probs))
        det_result.write(detailed_str + preprobs_str + "\n")

        if y_mask[t_count] == 0:
            # softmax of cluster and classifier
            clu_label_idx, max_idx, clu_softmax = cluster_prediction(cluster_models, dict_x_i, "inv")
            clf_softmax = softmax(classify_result)
            clf_label_idx = np.argmax(clf_softmax)

            clu_pseudo_label_ratio = class_ratio[clu_label_idx]
            clf_pseudo_label_ratio = class_ratio[clf_label_idx]

            ##### chose the explorer with higher gmean value
            gmean_classifier = np.prod(recall_classifier)
            gmean_cluster = np.prod(recall_cluster)
            sum_gmean = gmean_classifier + gmean_cluster

            if sum_gmean==0:
                prob_choose_classifier = 0.5
                prob_choose_cluster = 0.5
            else:
                prob_choose_classifier = gmean_classifier/sum_gmean
                prob_choose_cluster = gmean_cluster/sum_gmean

            # randomly exploration
            rand_chosen_value = random.random()
            if rand_chosen_value<=prob_choose_cluster:
                pseudo_label_ratio = clu_pseudo_label_ratio
                label_idx = clu_label_idx
            else:
                pseudo_label_ratio = clf_pseudo_label_ratio
                label_idx = clf_label_idx

            chosen_thre = 1 / (np.exp(k_inverse * len(class_exist) * pseudo_label_ratio))

            rand_value = random.random()

            if rand_value >= chosen_thre:
                label_idx = -1
            #####

            ##### when novel classes emerge, using cluster method to get pseudo-label to train classifier
            if label_idx != -1:
                pseudo_label = class_exist[label_idx]
                cluster_model = cluster_models[label_idx]

                [class_exist, class_ratio, class_ratio_initial, class_disap, class_rec] = classRatioUpdate(class_exist,
                                                                                                           class_ratio,
                                                                                                           class_ratio_initial,
                                                                                                           pseudo_label,
                                                                                                           disp_threshold)

                pseudo_subscript = np.take(np.argwhere(class_exist == pseudo_label), 0)

                classifier_count = len(ensemble)
                for i in range(len(ensemble)):
                    K = poisson.rvs(mu=1, size=M)
                    clf_class = class_exist[i]

                    if (i == pseudo_subscript):
                        label_tmp = 1
                    else:
                        ratio_tmp = class_ratio[i]

                        select_ratio = ratio_tmp / (1 - ratio_tmp)
                        label_tmp = -1
                        K = poisson.rvs(mu=select_ratio, size=M)

                    # updation info
                    if label_tmp == +1:
                        number_updation[clf_class]["positive"].append([t_count, real_label, np.mean(K)])
                    elif label_tmp == -1:
                        number_updation[clf_class]["negative"].append([t_count, real_label, np.mean(K)])

                    for idx_model, model in enumerate(ensemble[i].model_list):

                        generate_sample = cluster_generate(cluster_model, dict_x_i, new_example)

                        update_time = K[idx_model]

                        if update_time == 0:
                            continue

                        _, ft_value = model.classify(generate_sample)
                        (param, new_alpha, new_norm) = model.update(generate_sample, label_tmp, ft_value)
                        model.currentAlpha = param * model.currentAlpha
                        model.currentAlpha[0, model.index] = update_time * new_alpha
                        model.norm2X[0, model.index] = new_norm
                        model.trainFea[:, [model.index]] = generate_sample
                        model.index = model.index + 1
                        if model.index >= model.cnt:
                            model.index = 0
                            model.firstloop = 0


        else:
            [class_exist, class_ratio, class_ratio_initial, class_disap, class_rec] = classRatioUpdate(class_exist, class_ratio,
                                                                                                       class_ratio_initial,
                                                                                                       real_label,
                                                                                                       disp_threshold)


            real_subscript = np.take(np.argwhere(class_exist == real_label), 0)
            if real_subscript == len(ensemble):
                ensemble.append(Ensemble(OnlineKLR, M, args = (eta, lamda, kernel_t, cnt, dimension)))

                ##### initialize a cluster model for novel class
                cluster_models.append(auto_eps_denstream(n_samples_init=4, stream_speed=1, decaying_factor= cluster_decay_factor))
                #####

                ##### initialize recall for novel class
                recall_cluster.append(0)
                recall_classifier.append(0)
                #####

                number_updation[real_label] = {"positive":[], "negative":[]}

                if len(ft_array) == 0:
                    ft_array = np.zeros([M, 1])
                else:
                    ft_array = np.concatenate((ft_array, np.zeros([M, 1])), axis = 1)
            else:
                ##### update recall for each class
                recall_classifier[real_subscript] = ratio_decay * recall_classifier[real_subscript] + (
                            1 - ratio_decay) * (prediction == real_label)

                label_idx, max_idx, clu_softmax = cluster_prediction(cluster_models, dict_x_i, "inv")
                if label_idx is np.NaN:
                    recall_cluster[real_subscript] = ratio_decay * recall_cluster[real_subscript]
                else:
                    recall_cluster[real_subscript] = ratio_decay * recall_cluster[real_subscript] + (
                                1 - ratio_decay) * (label_idx == real_subscript)
                #####

            # Update the corresponding cluster model, only when true label is available
            cluster_models[real_subscript].learn_one(dict_x_i)

            classifier_count = len(ensemble)
            for i in range(len(ensemble)):
                K = poisson.rvs(mu=1, size = M)
                clf_class = class_exist[i]

                if (i == real_subscript):
                    label_tmp = 1
                else:
                    ratio_tmp = class_ratio[i]

                    select_ratio = ratio_tmp / (1 - ratio_tmp)
                    label_tmp = -1
                    K = poisson.rvs(mu=select_ratio, size=M)

                # updation info
                if label_tmp == +1:
                    number_updation[clf_class]["positive"].append([t_count, real_label, np.mean(K)])
                elif label_tmp == -1:
                    number_updation[clf_class]["negative"].append([t_count, real_label, np.mean(K)])

                for idx_model, model in enumerate(ensemble[i].model_list):
                    update_time = K[idx_model]

                    if update_time == 0:
                        continue

                    if np.isnan(ft_array[idx_model, i]):
                        (_, ft_array[idx_model, i]) = model.classify(new_example)

                    (param, new_alpha, new_norm) = model.update(new_example, label_tmp, ft_array[idx_model, i])
                    model.currentAlpha = param * model.currentAlpha
                    model.currentAlpha[0, model.index] = update_time * new_alpha
                    model.norm2X[0, model.index] = new_norm
                    model.trainFea[:, [model.index]] = new_example
                    model.index = model.index + 1
                    if model.index >= model.cnt:
                        model.index = 0
                        model.firstloop = 0

    fid_result.close()
    det_result.close()

    with open(updation_result, "wb") as file:
        pickle.dump(number_updation, file)

    duration = time.time() - begin_time
    pc_name = subprocess.run("hostname", capture_output=True, text=True).stdout.strip()
    savemat(time_result, {"time": duration, "pc_name": pc_name})

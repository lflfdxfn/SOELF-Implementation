import numpy as np
import matplotlib.pyplot as plt
import math
import warnings
from scipy.special import softmax


from river import cluster
from river import stream
from river.cluster import DenStream
from scipy.io import loadmat
from scipy.spatial import distance_matrix


class auto_eps_denstream(DenStream):
    def __init__(self, n_samples_init, stream_speed, decaying_factor):
        super().__init__(n_samples_init = n_samples_init,
                         stream_speed = stream_speed,
                         decaying_factor = decaying_factor)

    def _initial_dbscan(self):
        # convert to array(list) data type
        buffer_array = []
        for item in self._init_buffer:
            buffer_array.append(list(item.x.values()))

        # calculate distance matrix
        # the distance in the denstream of the river package is squared.
        dist_matrix = distance_matrix(buffer_array, buffer_array) ** 0.5
        len_buffer = len(buffer_array)

        # calculate mu-th dist for each example
        mu_th_dist_list = []
        for i in range(len_buffer):
            example_dist_list = np.sort(dist_matrix[i, :])

            other_dist_list = example_dist_list[1:] # first element is 0 (distance of an example to itself), drop it.
            not_found = 1
            for other_dist in other_dist_list:
                if sum(other_dist_list < other_dist) >= self.mu:
                    mu_th_dist_list.append(other_dist)

                    not_found = 0
                    break

            if not_found:
                mu_th_dist_list.append(other_dist_list[-1] * 1.0001)

        mean_mu_th_dist = np.mean(mu_th_dist_list)
        std_mu_th_dist = np.std(mu_th_dist_list)
        epsilon = mean_mu_th_dist + max(std_mu_th_dist, mean_mu_th_dist * 0.0001)

        self.epsilon = epsilon
        super()._initial_dbscan()

        if len(self.p_micro_clusters)==0:
            warnings.warn("Error! No cluster is found!")


def cluster_generate(cluster_model:DenStream, dict_x_i, x_i):
    x_shape = x_i.shape

    ##### Generate samples:
    potential_samples = []
    distances = []

    for _, micro_cluster in cluster_model.p_micro_clusters.items():
        center = micro_cluster.calc_center(cluster_model.timestamp)
        distance = cluster_model._distance(center, dict_x_i)

        center_x = np.zeros(x_shape)

        for key, value in center.items():
            center_x[key, 0] = value

        potential_sample = x_i + np.random.random() * (center_x - x_i)
        potential_samples.append(potential_sample)
        distances.append(distance)

    if 0 in distances:
        for sample, distance in zip(potential_samples, distances):
            if distance == 0:
                assemble_sample = sample
                break
    else:
        reverse_distances = [1 / distance for distance in distances]
        assemble_sample = np.zeros(x_shape)

        for i_loc, sample in enumerate(potential_samples):
            weight = reverse_distances[i_loc] / np.sum(reverse_distances)
            assemble_sample += weight * sample

    return assemble_sample


def cluster_prediction(cluster_models, dict_x_i, metric = "inv"):
    # calculate the distance to the nearest cluster in each class
    dist_class = {}
    for idx_class, cluster_model in enumerate(cluster_models):
        micro_clusters_dict = cluster_model.p_micro_clusters

        class_min_dist = math.inf

        for _, micro_cluster in micro_clusters_dict.items():
            center = micro_cluster.calc_center(cluster_model.timestamp)
            distance = cluster_model._distance(center, dict_x_i)

            if distance < class_min_dist:
                class_min_dist = distance

        dist_class[idx_class] = class_min_dist

    # calculate the softmax function
    # distance to softmax
    dist_label_idxs = list(dist_class.keys())
    dist_values = list(dist_class.values())

    if metric == "inv":
        # from dist to classification results
        # deal with zero error
        if 0 in dist_values:
            loc_0 = np.NaN
            for index, value in enumerate(dist_values):
                if value == 0:
                    loc_0 = index
            softmax_result = np.zeros(len(dist_values))
            softmax_result[loc_0] = 1
        else:
            class_prob = list(map(lambda x: 1 / x, dist_values))
            softmax_result = softmax(class_prob)
    elif metric == "minus":
        # from dist to classification results
        class_prob = list(map(lambda x: -x, dist_values))

        # softmax function
        softmax_result = softmax(class_prob)

    # softmax to classification and updation factor
    max_idx = np.argmax(softmax_result)
    label_idx = dist_label_idxs[max_idx]

    return label_idx, max_idx, softmax_result
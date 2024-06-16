import os
import numpy as np

from scipy.io import loadmat

class data_params:
    parameters = {
        # eta, lamda, t, decay factor
        "letter": [9, 0.0001, 5.29, 0.9],
        "statlog": [3, 0.0001, 32.7, 0.9],
        "mnist": [3, 0.0001, 4.61, 0.9],
        "covertype": [1, 0.0001, 4.0, 0.9],
        "laden_ce": [8, 0.000100, 1.238610, 0.9],
        "christ_ce": [3, 0.000100, 1.249509, 0.9],
        "wed_ce": [5, 0.000300, 1.243258, 0.9],
        "huge_tweet": [2, 0.000100, 1.317914, 0.9],
        "kddcup99": [0.000100, 0.000100, 0.295834, 0.9],
        "pokerlsn": [8, 0.001000, 3.366803, 0.9]
    }

    def __init__(self, run_data, settings:dict):
        # data file path
        self.name = run_data
        data_dir = settings["data_path"]
        self.data_path = os.path.join(data_dir, self.name + ".mat")

        # data and data info
        data = loadmat(self.data_path)
        self.x = data["x"].astype(float)
        self.y = np.squeeze(data["y"]).astype(float)
        (self.data_n_dim, self.data_n_cnt) = self.x.shape
        self.data_n_classes = len(np.unique(self.y))

        # class appear time
        self.data_appear_time = np.zeros(self.data_n_classes, dtype = int)
        for idx_cls in range(1, self.data_n_classes+1):
            self.data_appear_time[idx_cls - 1] = np.take(np.argwhere(self.y==idx_cls), 0)

        # other parameters
        self.disp_threshold = 1e-5
        
        # use corresponding parameters
        self.eta = None
        self.lamda = None
        self.t = None
        self.decay_factor = None
        for key, value in data_params.parameters.items():
            if key in self.name:
                (self.eta, self.lamda, self.t, self.decay_factor) = value
                break

    def print_info(self):
        # print data info
        print("Dataset: {}".format(self.name))
        print("\tdata path: {}".format(self.data_path))
        print("\tn_examples: {}".format(self.data_n_cnt))
        print("\tn_dim: {}".format(self.data_n_dim))
        print("\tn_classes: {}".format(self.data_n_classes))
        print("\tappear time (in python index): {}".format(str(self.data_appear_time)))
        
        # print parameters chosen for KLR
        print("Parameters Choose for Online KLR:")
        print("\teta: {}".format(self.eta))
        print("\tlamda: {}".format(self.lamda))
        print("\tsigma: {}".format(self.t))
        print("\tdecay factor: {}".format(self.decay_factor))
        print("\tdisappearing threshold: {}".format(self.disp_threshold))
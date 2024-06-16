import numpy as np

from scipy.io import loadmat

class weakly_settings:

    def __init__(self, weakly_scenarios:dict):
        self.init_index = weakly_scenarios["init_index"]
        self.m = weakly_scenarios["m"]
        self.p = weakly_scenarios["p"]
        self.mask_path = weakly_scenarios["mask_path"]

    def print_info(self):
        print("Setting of Weakly Scenarios:")
        print("\tInit_index: {}".format(self.init_index))
        print("\tm: {}".format(self.m))
        print("\tp: {:.2f}".format(self.p))

    def weakly_mask(self, data_name:str):
        mask_file_path = "{}/{}_{}_{:.2f}/{}_1.mat".format(self.mask_path, self.init_index, self.m, self.p, data_name);
        data = loadmat(mask_file_path);
        y_mask = np.squeeze(data["y_mask"].astype(float))

        return y_mask
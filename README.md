# SOELF

## Dependencies
* Python (in `requirements.txt`):
  * python==3.8.8
  * matplotlib==3.3.4
  * numpy==1.24.4
  * pandas==2.0.3
  * river==0.19.0
  * scikit-learn==0.24.1
  * scipy==1.6.2
* MATLAB R2022b

## Data Preparation
### Download Data Streams Mentioned in the Paper
* Download from [this link](https://drive.google.com/file/d/1NBOTIRT7QrE-xXBZMizEx9qX1zjd3fs_/view?usp=drive_link). Unzip and put into the direcroty `DataStream`.
### Prepare Your Own Data Streams
1. Class Emergence:
   * Put your offline datasets in `offline_datasets`. The class label should be in the last column, starting from `1`.
   * Setting parameters in `Synthetic_Emergence.py`:
     * Create a new data loading path and add it into the dictionary `datasets`.
     * `n_scenario`: the number of scenario to annotate the data stream;
     * `n_existing`: the number of existing classes;
     * `n_emerging`: the number of emerging classes;
     * `mean_points`: the location (by chunk) where the emerging classes reaches maximum prior probability;
     * `max_values`: the value of the maximum prior probability;
     * `disp_or_nots`: whether emerging classes disappear later or not;
     * `e_durations`: the duration (by chunk) for which the emerging classes persist at a maximum prior probability;
     * `n_chunk`: the overall lengh (by_chunk) of the data stream;
     * `chunk_size`: the lengh of each chunk;
     * `seed`: random seed;
     * `gaussian_std`: describes the increase ratio of the prior probability of emerging classes;
     * `emer_threshold`: the threshold of the prior probability of a class to be regarded as an existing class;
     * `folder_name`: the output folder.
   * Run `Synthetic_Emergence.py`.
2. Class Disappearaence and Reoccurrence:
   * Setting parameters in `Synthetic_DispReoccur.py`:
     * Create a new data loading path and add it into the dictionary `datasets`.
     * `n_scenario`: the number of scenario to annotate the data stream (Should be different from the number in class emergence);
     * `n_existing`: the number of existing classes;
     * `disp_point`: the location (by chunk) where an existing class disappear;
     * `zero_dura`: the duration (by chunk) of an disappeared classes has a zero prior probability;
     * `reoccur_dura`: the duration (by chunk) of an reoccurred classes has a non-zero prior probability;
     * `chunk_size`: the length of each chunk;
     * `seed`: random seed;
     * `gaussian_std`: describes the increase/decrease ratio of the prior probability of the disappeared and reoccurred class;
     * `folder_name`: the output folder.
   * Run `Synthetic_DispReoccur.py`.
3. Incomplete Supervision:
   * Setting Parameters in `label_mask_synthetic.m` and `real_mask_real_world.m`:
     * `origin_data`: the name of the original offline datasets or real-world data streams;
     * `scenarios`: choices of the number annotated the scenarios of your synthetic data streams;
     * `n_cases`: choices of the correponding case of each scenario;
     * `init_indexes`: choices of the size of initial training dataset;
     * `weakly_m`: choices of parameter $l$;
     * `weakly_p`: choices of parameter $pr$;
     * `data_path`: where to load data streams;
     * `output_path`: where to store label mask files.
   * Run `label_mask_synthetic.m` and `real_mask_real_world.m`.


## Experiments

## Useful Information
### Files of Results
### Parameter Settings
clc;clear;

n_runs = 1;

%%% gen_dataset
origin_data = ["huge_tweet", "kddcup99", "pokerlsn", "laden_ce", "christ_ce", "wed_ce"];

init_indexes = [1000];
weakly_m = [10, 100];
weakly_p = [0, 0.05, 0.1];
data_path = "./";
output_path = "./label_mask";

data_names = [];

data_names = origin_data;

for idx_init = 1:numel(init_indexes)
    init_index = init_indexes(idx_init);

    for idx_m = 1:numel(weakly_m)
        for idx_p = 1:numel(weakly_p)
            m = weakly_m(idx_m);
            p = weakly_p(idx_p);
            
            for idx_data = 1:numel(data_names)
                
                for i_run = 1:n_runs

                    data_name = data_names(idx_data);
                    file_path = sprintf("%s/%s.mat", data_path, data_name);
                    x = load(file_path,'x');
                    y = load(file_path,'y');
                    x = x.x;
                    y = y.y;    
                    
                    rng(i_run, 'Threefry');
                    y_mask = zeros(numel(y), 1);
    
                    % init is all known
                    y_mask(1:init_index) = 1;
                
                    % m
                    for idx_cls = 1:numel(unique(y))
                        y_mask(find(y==idx_cls, m)) = 1;
                    end
    
                    % p
                    rand_numbers = rand(1, numel(y));
                    chosen_idx = rand_numbers<=p;
                    y_mask(chosen_idx) = 1;
    
                    % store this weakly mask
                    output_dir = sprintf("%s/%d_%d_%.2f", output_path, init_index, m, p);
                    
                    if ~exist(output_dir, "dir")
                        mkdir(output_dir)
                    end
    
                    output_file = sprintf("%s/%s_%d.mat", output_dir, data_name, i_run);
                    save(output_file, 'y_mask', 'y', 'init_index', 'm', 'p')
                end
            end
        end
    end
end

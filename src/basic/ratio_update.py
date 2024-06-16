import numpy as np

def classRatioUpdate(o_class_exist, o_class_ratio, o_class_ratio_initial, current_class_label, disp_threshold, ratio_decay = 0.9):

    current_class_subscript = np.argwhere(np.array(o_class_exist)==current_class_label)
    if len(current_class_subscript) == 0:
        current_class_subscript = None
    else:
        current_class_subscript = np.take(current_class_subscript, 0)

    class_count = len(o_class_exist)
    class_disap = []
    class_rec = -1

    class_exist = o_class_exist
    class_ratio = o_class_ratio
    class_ratio_initial = o_class_ratio_initial

    # update for the class that current example belongs to
    if current_class_subscript is None:
        # novel class emergence
        current_class_subscript = class_count
        class_exist = np.append(class_exist, current_class_label)
        class_ratio = np.append(class_ratio, 0)
        class_ratio_initial = np.append(class_ratio_initial, 1)
    elif class_ratio[current_class_subscript] == 0:
        # recurrent class + second example needed for calculate ratio (recurrent or novel)
        if class_ratio_initial[current_class_subscript] == 0:
            # first reveive (recurrent)
            class_rec = current_class_subscript
            class_ratio_initial[current_class_subscript] = 1
        else:
            # second receive (recurrent or novel)
            new_ratio = 1/class_ratio_initial[current_class_subscript]
            class_ratio[current_class_subscript] = new_ratio
            class_ratio = class_ratio/sum(class_ratio)
            class_ratio[current_class_subscript] = ratio_decay * class_ratio[current_class_subscript] + 1 - ratio_decay
            class_ratio_initial[current_class_subscript] = 0
    else:
        # current exisiting class
        class_ratio[current_class_subscript] = ratio_decay*class_ratio[current_class_subscript]+1-ratio_decay

    # update for the other classes
    for j in range(class_count):
        if current_class_subscript != j:
            # update ratio initial count
            if class_ratio_initial[j] != 0:
                class_ratio_initial[j] = class_ratio_initial[j]+1

            # update ratio percentage
            if class_ratio[j] != 0:
                class_ratio[j] = ratio_decay * class_ratio[j]
                # set class disappearence
                if class_ratio[j] < disp_threshold:
                    class_ratio[j] = 0
                    class_ratio_initial[j] = 0
                    class_disap.append(j)

    return class_exist, class_ratio, class_ratio_initial, class_disap, class_rec
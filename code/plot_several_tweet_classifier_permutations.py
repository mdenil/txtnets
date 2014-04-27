__author__ = 'mdenil'

import numpy as np
import matplotlib.pyplot as plt
import os
import cPickle as pickle

models_dir = "models"

if __name__ == "__main__":
    fig_words = plt.figure()
    fig_layers = plt.figure()
    fig_embsize = plt.figure()

    for model_file_name in os.listdir(models_dir):
        with open(os.path.join(models_dir, model_file_name)) as model_file:
            model_info = pickle.load(model_file)

        acc = [(info['batch_index'], info['acc']) for info in model_info['monitor_info']]
        cost = [(i, info['cost']) for i,info in enumerate(model_info['iteration_info'])]
        acc_x, acc_y = zip(*acc)
        cost_x, cost_y = zip(*cost)

        fig_words.gca().plot(acc_x, acc_y, 'r' if 'words' in model_file_name else 'b')
        fig_layers.gca().plot(acc_x, acc_y, 'r' if 'one_layer' in model_file_name else 'b')
        fig_embsize.gca().plot(acc_x, acc_y, 'r' if 'large' in model_file_name else 'b')

    fig_words.gca().set_title("Words vs characters")
    fig_layers.gca().set_title("One layer vs two layer")
    fig_embsize.gca().set_title("Large embedding vs small embedding")

    plt.show()
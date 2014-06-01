__author__ = 'albandemiraj'




__author__ = 'albandemiraj'
import cPickle as pickle
import os
import yaml
import pandas as pd
import argparse




def run():
    parser = argparse.ArgumentParser(
        description="Collect cluster data.")
    parser.add_argument("--progress_dir", help="where the results subset of folders is")
    args = parser.parse_args()

    #Preparing filenames
    progress_dir = args.progress_dir

    list_of_dictionaries = []

    #START
    for result_folder in os.listdir(progress_dir):
        if result_folder == '.DS_Store':
            continue

        #LOADING DICTIONARIES
        try:
            with open(os.path.join(progress_dir, result_folder, 'progress.pkl')) as file:
                progress_dict = pickle.load(file)
        except IOError:
            print "Failed to load progress.pkl from {}".format(result_folder)
            continue

        try:
            with open(os.path.join(progress_dir, result_folder, 'params_train.yaml'), "r") as stream:
                params_train_dict = yaml.load(stream)
        except IOError:
            print "Failed to load params_train.yaml from {}".format(result_folder)
            continue

        try:
            with open(os.path.join(progress_dir, result_folder, 'params_experiment_config.yaml'), "r") as stream:
                params_experiment_dict = yaml.load(stream)
        except IOError:
            print "Failed to load params_experiment_config.yaml from {}".format(result_folder)
            continue

        #FLATTENING DICTIONARY
        new_dictionary = dict()
        for key in params_experiment_dict:
            count = 1
            if key == 'word_layers':
                for inner_dict in params_experiment_dict[key]:
                    for key in inner_dict:
                        new_dictionary[key+'_w_'+str(count)]=inner_dict[key]
                    count += 1
            elif key == 'sentence_layers':
                for inner_dict in params_experiment_dict[key]:
                    for key in inner_dict:
                        new_dictionary[key+'_s_'+str(count)]=inner_dict[key]
                    count += 1
            else:
                new_dictionary[key]=params_experiment_dict[key]

        params_experiment_dict = new_dictionary

        #PUTTING DICTIONARIES TOGATHER
        for batch_dict in progress_dict:
            temp_dict = dict()
            temp_dict.update(batch_dict)
            temp_dict.update(params_experiment_dict)
            temp_dict.update(params_train_dict)
            list_of_dictionaries.append(temp_dict)


    #GETTING IT INTO PANDA
    data = pd.DataFrame(list_of_dictionaries)

    with open("pandas_data.pkl", 'w') as file:
       pickle.dump(data, file, protocol=-1)

if __name__ == "__main__":
    run()

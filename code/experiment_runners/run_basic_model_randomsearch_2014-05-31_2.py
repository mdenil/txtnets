__author__ = 'mdenil'


import os
import random
import cluster.job_manager
from cluster.config_utils import product, expand


def _sentence_embedding_size(k_max, n_feature_maps):
    return k_max * n_feature_maps


def _softmax_input_dimensions(k_max, n_feature_maps):
    return k_max * n_feature_maps


def make_job_template(
        embedding_dimension,
        n_feature_maps_w1,
        n_feature_maps_w2,
        n_feature_maps_d1,
        kernel_width_w1,
        kernel_width_w2,
        kernel_width_d1,
        k_max_w1,
        k_max_w2,
        k_max_d1,
        n_epochs,
        adagrad_gamma,
        validation_frequency,
        save_frequency,
        batch_size,
        walltime,
        n_classes,
        pmem,
        ppn,
        dropout,
        regularizer,
        ):
    return {
        "templates": [
            {
                "target": "launcher.sh",
                "params_target": "params_launcher.yaml",
                "src": "launcher.sh",
                "params": {
                    "code_root": os.path.abspath(os.environ['HERE']),
                    "walltime": walltime,
                    "pmem": pmem,
                    "ppn": ppn,
                },
            },
            {
                "target": "train.py",
                "params_target": "params_train.yaml",
                "src": "train.py",
                "params": {
                    "train_data_json": os.path.abspath(
                        os.path.join(
                            os.environ['DATA'],
                            "stanfordmovie",
                            "stanfordmovie.train.sentences.clean.projected.json")),
                    "train_encoding_json": os.path.abspath(
                        os.path.join(
                            os.environ['DATA'],
                            "stanfordmovie",
                            "stanfordmovie.train.sentences.clean.dictionary.encoding.json")),
                    "n_validation": 500,
                    "batch_size": batch_size,
                    "fixed_n_sentences": 30,
                    "fixed_n_words": 50,
                    "regularizer": regularizer,
                    "adagrad_gamma": adagrad_gamma,
                    "n_epochs": n_epochs,
                    "validation_frequency": validation_frequency,
                    "save_frequency": save_frequency,
                },
            },
            {
                "target": "experiment_config.py",
                "params_target": "params_experiment_config.yaml",
                "src": "basic_model.py",
                "params": {
                    "embedding_dimension": embedding_dimension,
                    "word_layers": [
                        {
                            # first word layer
                            "n_feature_maps": n_feature_maps_w1,
                            "kernel_width": kernel_width_w1,
                            "n_channels": embedding_dimension,
                            "k_pooling": k_max_w1,
                            "k_dynamic": 0.5,
                            "nonlinearity": "Tanh",
                        },
                        {
                            # second word layer
                            "n_feature_maps": n_feature_maps_w2,
                            "kernel_width": kernel_width_w2,
                            "n_channels": n_feature_maps_w1,
                            "k_pooling": k_max_w2,
                            "k_dynamic": -1,
                            "nonlinearity": "Tanh",
                        }
                    ],
                    "sentence_layers": [
                        {
                            "n_feature_maps": n_feature_maps_d1,
                            "kernel_width": kernel_width_d1,
                            "n_channels": _sentence_embedding_size(
                                k_max=k_max_w2,
                                n_feature_maps=n_feature_maps_w2),
                            "k_pooling": k_max_d1,
                            "k_dynamic": -1,
                            "nonlinearity": "Tanh",
                        }
                    ],
                    "n_classes": n_classes,
                    "dropout": dropout,
                    "softmax_input_dimensions": _softmax_input_dimensions(
                        k_max=k_max_d1,
                        n_feature_maps=n_feature_maps_d1),
                },
            }
        ],

        "task_params": {
            "launcher_file": "launcher.sh"
        },
    }


if __name__ == "__main__":
    random.seed(23233)

    experiment_dir = os.path.abspath(os.path.join(
        os.environ['RESULTS'],
        "basic_model_randomsearch_2014-05-31_2"))

    parameter_search_grid_full = product([
        expand({
            "embedding_dimension": 20
        }),
        expand({
            "n_feature_maps_w1": range(5, 12+1, 1),
            "n_feature_maps_w2": range(20, 30+1, 1),
            "n_feature_maps_d1": range(20, 30+1, 1),
        }),
        expand({
            "kernel_width_w1": range(5, 15+1),
            "kernel_width_w2": range(5, 10+1),
            "kernel_width_d1": range(5, 15+1),
        }),
        expand({
            "k_max_w1": 7,
            "k_max_w2": 5,
            "k_max_d1": 5,
        }),
        expand({
            "regularizer": [1e-4, 5e-3, 1e-3],
            "dropout": True,
            "n_epochs": 100000,
            "adagrad_gamma": 0.01,
            "validation_frequency": 50,
            "save_frequency": 50,
            "batch_size": 50,
            "walltime": "24:00:00",
            "n_classes": 2,
            "pmem": "6gb",
            "ppn": 2,
        })
    ])

#    def is_valid_config(config):
#        valid = True
#        valid = valid and config['k_max_w1'] >= config['k_max_w2']
#        return valid
#
#    parameter_search_grid_full = filter(is_valid_config, parameter_search_grid_full)

    print len(parameter_search_grid_full)

    random.shuffle(parameter_search_grid_full)
    parameter_search_grid = random.sample(parameter_search_grid_full, 300)

    job_templates = [make_job_template(**params) for params in parameter_search_grid]

    jobs = []
    for job_id, task in enumerate(job_templates):
        jobs.append(cluster.job_manager.Job(
            job_id=job_id,
            base_dir=experiment_dir,
            params={},
            template_dir=os.path.join(os.environ['ROOT'], "experiment_templates"),
            tasks=[task],
            task_factory=cluster.job_manager.ClusterTask))

    # # shuffle so that large tasks are spread out approximately uniformly
    random.shuffle(jobs)

    for job in jobs:
        for task in job.tasks():
            task.configure()
            task.launch()

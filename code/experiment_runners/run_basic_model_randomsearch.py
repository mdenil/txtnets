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
                    "fixed_n_sentences": 15,
                    "fixed_n_words": 50,
                    "regularizer": 1e-4,
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
    random.seed(2323423)

    experiment_dir = os.path.abspath(os.path.join(
        os.environ['RESULTS'],
        "test_job_launcher"))

    parameter_search_grid_full = product([
        expand({
            "embedding_dimension": 30
        }),
        expand({
            "n_feature_maps_w1": range(5, 25+1, 3),
            "n_feature_maps_w2": range(5, 25+1, 3),
            "n_feature_maps_d1": range(5, 25+1, 3),
        }),
        expand({
            "kernel_width_w1": range(3, 7),
            "kernel_width_w2": range(3, 7),
            "kernel_width_d1": range(3, 7),
        }),
        expand({
            "k_max_w1": range(2, 8, 2),
            "k_max_w2": range(2, 6, 2),
            "k_max_d1": range(2, 4),
        }),
        expand({
            "dropout": [True, False],
            "n_epochs": 10000,
            "adagrad_gamma": 0.01,
            "validation_frequency": 50,
            "save_frequency": 50,
            "batch_size": 50,
            "walltime": "10:00:00",
            "n_classes": 2,
            "pmem": "4gb",
            "ppn": 4,
        })
    ])

    def is_valid_config(config):
        valid = True
        valid = valid and config['k_max_w1'] >= config['k_max_w2']
        return valid

    parameter_search_grid_full = filter(is_valid_config, parameter_search_grid_full)

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
            task_factory=cluster.job_manager.LocalTask))

    # # shuffle so that large tasks are spread out approximately uniformly
    random.shuffle(jobs)

    for job in jobs:
        for task in job.tasks():
            task.configure()
            task.launch()

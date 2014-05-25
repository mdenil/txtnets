__author__ = 'mdenil'

import os
import subprocess
import itertools
from datetime import datetime


def time_stamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

run_cmd_template = "bash run.sh python -u train_multilingual_embeddings.py {} | tee {}"

results_dir = "../results"

batch_sizes = [100, 200]
learning_rates = [0.05]
n_sampless = [50]
margins = [40]
embedding_dimensions = [40]
regularizers = [1.0]
max_epochss = [5]

space = itertools.product(
    batch_sizes,
    learning_rates,
    n_sampless,
    margins,
    embedding_dimensions,
    regularizers,
    max_epochss)

run_folder = os.path.join(results_dir, "run_{}".format(time_stamp()))

os.mkdir(run_folder)

for bs, lr, ns, m, ed, r, me in space:
    save_file_stem = "model_bs{}_lr{}_ns{}_m{}_ed{}_r{}_me{}".format(
        bs, lr, ns, m, ed, r, me)
    save_file_name = os.path.join(run_folder, save_file_stem + ".pkl")
    output_file_name = os.path.join(run_folder, save_file_stem + ".log")

    params = " ".join(map(str, [
        "--batch_size", bs,
        "--learning_rate", lr,
        "--n_samples", ns,
        "--margin", m,
        "--embedding_dimension", ed,
        "--regularizer", r,
        "--max_epochs", me,
        "--save_file_name", save_file_name,
        "--results_dir", ".",
    ]))

    run_cmd = run_cmd_template.format(params, output_file_name)

    subprocess.call(run_cmd, shell=True)
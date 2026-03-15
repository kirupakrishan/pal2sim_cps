import random

import numpy as np
import pandas as pd

from utils.config import Config
from data_handler import DataHandler
from classificators.dummy_classifier import DummyClassifier
from classificators.random_forest_classifier import RandomForestClassifierSK
from utils.utils import calculate_mcc_multilabel, plot_per_class_confusion

if __name__ == '__main__':

    config = Config()

    # Seeding
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    # if you use any other libraries that require seeding, set it here as well (e.g., torch.manual_seed(SEED) for PyTorch)
    # -> your results should be reproducible across runs with the same seed


    val_mccs = []
    test_mccs = []
    lr_histories_by_fold = {}

    # load data
    datahandler = DataHandler(config=config)

    # Leave-one-out: EXPERIMENT_ID = 1..4
    for fold in range(1, 5):
        print(f"\n--- Fold {fold}/4 | EXPERIMENT_ID={fold} ---")
        val_id = fold + 1 if fold < 4 else 1

        datahandler.config.data.test_experiment_id = fold
        # validation hat to be different from test
        datahandler.config.data.validation_experiment_id = val_id

        train, val, test, target_vals = datahandler.get_data_loaders()
        X_train, y_train = train
        X_val, y_val = val
        X_test, y_test = test

        print(X_train[0][0][0])
        print(y_train.shape)


import csv
import os


def save_dataset(data, label, name, path, opt_prefix, opt_suffix):
    """
    A method that save dataset with optional prefix and suffix
    :param data: data column of dataset
    :param label: label column of dataset
    :param name: name of the new dataset like train, test or validation dataset
    :param path: desired path for saving
    :param opt_prefix: optional prefix
    :param opt_suffix: optional suffix
    :return:
    """
    # Saving a text file to train/test the classifier
    os.makedirs(path, exist_ok=True)
    dir_train = os.path.join(path, f"{opt_prefix}_{name}_{opt_suffix}.txt")
    with open(dir_train, "w") as fwt:
        fwt.write("\n".join([" ".join([label, data]) for data, label in zip(data, label)]))


def normalize_and_save_dataset(train_data, test_data, train_path, test_path, opt_prefix="", opt_suffix=""):
    """
    A method that normalize and save dataset for fasttext
    :param train_data: 
    :param test_data:
    :param train_path:
    :param test_path:
    :param opt_prefix:
    :param opt_suffix:
    :return:
    """
    # Saving a text file to train/test the classifier
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    train_data = train_data.to_csv(os.path.join(train_path, f"{opt_prefix}_train_{opt_suffix}.txt"),
                                   index=False,
                                   sep=' ',
                                   header=None,
                                   quoting=csv.QUOTE_NONE,
                                   quotechar="",
                                   escapechar=" ")

    test_data = test_data.to_csv(os.path.join(test_path, f"{opt_prefix}_test_{opt_suffix}.txt"),
                                 index=False,
                                 sep=' ',
                                 header=None,
                                 quoting=csv.QUOTE_NONE,
                                 quotechar="",
                                 escapechar=" ")

    return train_data, test_data

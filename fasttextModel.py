import os
import pandas as pd
import fasttext
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from getconfig import get_config
from addprefix import change_labels
from savedataset import normalize_and_save_dataset


class Model:
    def __init__(self, path=get_config("fasttext")["model_dir"],
                 suffix=get_config("fasttext")["suffix"],
                 prefix=get_config("fasttext")["prefix"],
                 tokenizer_model=get_config("fasttext")["tokenizer_model"],
                 train_file=get_config("fasttext")["train_file"],
                 valid_file=get_config("fasttext")["valid_file"]):

        self.dir_root = os.path.dirname(os.getcwd())
        self.dir_model_root = os.path.join(self.dir_root, "Data", "Models")
        os.makedirs(self.dir_model_root, exist_ok=True)

        self.model = None
        self.is_trained = False
        self.path = path

        self.suffix = suffix
        self.prefix = prefix
        self.tokenizer_model = tokenizer_model
        self.dir_model_save = os.path.join(self.dir_model_root,
                                           f"{self.prefix}_model_with_{self.tokenizer_model}_{self.suffix}")
        self.train_file = train_file
        self.valid_file = valid_file

        if self.path is not None:
            self.load_pretrained()
        else:
            self.train_supervised_model()

    def train_supervised_model(self, autotune_mode=get_config("fasttext")["autotune_mode"],
                               autotuneduration=get_config("fasttext")["autotuneduration"],
                               lr=get_config("fasttext")["lr"],
                               epoch=get_config("fasttext")["epoch"],
                               wordNgrams=get_config("fasttext")["wordNgrams"],
                               opt_save=get_config("fasttext")["opt_save"]):
        """
        A method that trains supervised model according to input parameters in config.json
        :return:
        """

        if self.path is not None:
            print(f"Loading model from: {self.path}")
            self.load_pretrained()

        if autotune_mode:
            print("Training supervised model with autotune mode...")
            self.model = fasttext.train_supervised(self.train_file,
                                                   autotuneValidationFile=self.valid_file,
                                                   autotuneDuration=autotuneduration)
        else:
            print("Training supervised model with given parameters...")
            self.model = fasttext.train_supervised(self.train_file, lr, epoch, wordNgrams)

        if opt_save:
            print(f"Model saved into {self.dir_model_save}")
            self.model.save_model(self.dir_model_save)

    def validation(self):
        """
        A method that test the model with valid/test file
        :return:
        """
        print("Model validation... ")
        return self.model.test(self.valid_file)

    def predict_list(self, value):
        """
        A method that predict the result of input value
        :param value:
        :return:
        """
        print("Model prediction...")
        return [self.model.predict(item)[0][0] for item in value]

    def predict_value(self, value):
        return self.model.predict(value)[0][0]

    def load_pretrained(self):
        self.model = fasttext.load_model(self.path)

    def report(self, y_true, y_pred):
        """
        A method that reports model evaluation
        :param y_true:
        :param y_pred:
        :return:
        """
        dir_report = os.path.join(self.dir_root, "Data", "report.txt")
        print(f"Model report is written into {dir_report}")
        print(classification_report(y_true, y_pred))
        with open(dir_report, "a") as report:
            report.write("\n############ " + self.prefix + " Model " + "with " + self.tokenizer_model + "  "
                         + self.suffix + " ############\n" + classification_report(y_true, y_pred))


if __name__ == '__main__':
    # Read data from file
    data = pd.read_csv("/path/to/data")

    x_train, x_test, y_train, y_test = train_test_split(data["data"], data["label"], test_size=0.30, random_state=42)
    ytrain = change_labels(y_train)
    ytest = change_labels(y_test)

    train_df = pd.concat([x_train, ytrain], axis=1)
    test_df = pd.concat([x_test, ytest], axis=1)

    normalize_and_save_dataset(train_data=train_df, test_data=test_df, train_path="Data/Processed/train",
                               test_path="Data/Processed/test",
                               opt_prefix="000", opt_suffix="000")

    # Create classifier model
    model = Model(suffix="000", train_file="Data/Processed/train/000_train_000.txt")

    # Train supervised model in autotune mode using validation data
    model.train_supervised_model(autotune_mode=False)

    y_pred_data = model.predict_list(x_test)
    print(y_pred_data)

    model.report(y_true=ytest, y_pred=y_pred_data)

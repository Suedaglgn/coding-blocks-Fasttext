def change_labels(labels):
    """
    A method that adds prefix to label column for fasttext standards
    :param labels: label column
    :return: prefixed label column
    """
    return ["__label__" + item for item in labels]

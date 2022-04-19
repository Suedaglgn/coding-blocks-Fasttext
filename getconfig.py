import json


def get_config(tokenizer):
    with open("config.json", "r") as outfile:
        f = outfile.read()
    conf = json.loads(f)
    outfile.close()
    return conf[tokenizer]

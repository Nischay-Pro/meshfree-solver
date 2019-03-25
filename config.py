import json

def getConfig():
    with open("config.json","r") as f:
        config = json.load(f)
    return config

def save_obj(obj, name):
    with open(name + '.json', 'w') as f:
        json.dump(obj, f)

def load_obj(name):
    with open(name + '.json', 'r') as f:
        return json.load(f)
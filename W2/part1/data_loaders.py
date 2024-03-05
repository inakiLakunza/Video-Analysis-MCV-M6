import json


def load_sequential_hold_out(t="train"):
    if t == "train":
        with open("train_first.json", 'r') as file:
            train = json.load(file)
        return train
    elif t == "val":
        with open("val_first.json", 'r') as file:
            val = json.load(file)
        return val

    else:
        raise Exception("the kind of training must be train or val")
    
def load_random_data(t="train"):
    if t == "train":
        with open("train_random.json", 'r') as file:
            train = json.load(file)
        return train
    elif t == "val":
        with open("val_random.json", 'r') as file:
            val = json.load(file)
        return val
    else:
            raise Exception("the kind of training must be train or val")

def load_13(t="train"):
    if t == "train":
        with open("./fold2.json", 'r') as file:
            train = json.load(file)
        return train
    elif t == "val":
        with open("./fold1+3.json", 'r') as file:
            val = json.load(file)
        return val
    else:
            raise Exception("the kind of training must be train or val")

def load_12(t="train"):
    if t == "train":
        with open("./fold3.json", 'r') as file:
            train = json.load(file)
        return train
    elif t == "val":
        with open("./fold1+2.json", 'r') as file:
            val = json.load(file)
        return val
    else:
            raise Exception("the kind of training must be train or val")

def load_23(t="train"):
    if t == "train":
        with open("./fold1.json", 'r') as file:
            train = json.load(file)
        return train
    elif t == "val":
        with open("./fold3+2.json", 'r') as file:
            val = json.load(file)
        return val

    else:
            raise Exception("the kind of training must be train or val")

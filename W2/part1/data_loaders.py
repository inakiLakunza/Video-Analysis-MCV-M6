import json


def load_sequential_hold_out(t="train"):
    if t == "train":
        with open("datafolds/train_first.json", 'r') as file:
            train = json.load(file)
        return train
    elif t == "val":
        with open("datafolds/val_first.json", 'r') as file:
            val = json.load(file)
        return val

    else:
        raise Exception("the kind of training must be train or val")
    
def load_random_data(t="train"):
    if t == "train":
        with open("datafolds/train_random.json", 'r') as file:
            train = json.load(file)
        return train
    elif t == "val":
        with open("datafolds/val_random.json", 'r') as file:
            val = json.load(file)
        return val
    else:
            raise Exception("the kind of training must be train or val")

def load_13(t="train"):
    if t == "val":
        with open("datafolds/val_2.json", 'r') as file:
            train = json.load(file)
        return train
    elif t == "train":
        with open("datafolds/train_13.json", 'r') as file:
            val = json.load(file)
        return val
    else:
            raise Exception("the kind of training must be train or val")

def load_12(t="train"):
    if t == "val":
        with open("datafolds/val_3.json", 'r') as file:
            train = json.load(file)
        return train
    elif t == "train":
        with open("datafolds/train_12.json", 'r') as file:
            val = json.load(file)
        return val
    else:
            raise Exception("the kind of training must be train or val")

def load_23(t="train"):
    if t == "val":
        with open("datafolds/val_1.json", 'r') as file:
            train = json.load(file)
        return train
    elif t == "train":
        with open("datafolds/train_32.json", 'r') as file:
            val = json.load(file)
        return val

    else:
            raise Exception("the kind of training must be train or val")

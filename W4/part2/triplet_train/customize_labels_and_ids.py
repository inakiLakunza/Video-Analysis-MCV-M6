import os
import sys

import pickle


if __name__ == '__main__':

    PATH_PKL_S01 = "/ghome/group07/test/W4/part2/triplet_train/pickles/S01/info.pkl"
    PATH_PKL_S03 = "/ghome/group07/test/W4/part2/triplet_train/pickles/S03/info.pkl"
    PATH_PKL_S04 = "/ghome/group07/test/W4/part2/triplet_train/pickles/S04/info.pkl"

    with open(PATH_PKL_S01, "rb") as f:
        info_pickle_S01 = pickle.load(f)

    with open(PATH_PKL_S04, "rb") as f:
        info_pickle_S04 = pickle.load(f)

    with open(PATH_PKL_S03, "rb") as f:
        info_pickle_S03 = pickle.load(f)

    
    test_emb_list = []
    test_emb_dict = {}
    
    #print(info_pickle_S03)

    for track_label, imgs_list in info_pickle_S03.items():
        test_emb_list.extend(imgs_list)

        for img in imgs_list:
            img_name = img[:-4]
            test_emb_dict[img_name] = [int(track_label)]


    print(test_emb_list)
    print("\n\n\n")
    print(test_emb_dict)

    train_emb_list = []
    train_emb_dict = {}
    for track_label, imgs_list in info_pickle_S01.items():
        train_emb_list.extend(imgs_list)

        for img in imgs_list:
            img_name = img[:-4]
            train_emb_dict[img_name] = [int(track_label)]
    for track_label, imgs_list in info_pickle_S04.items():
        train_emb_list.extend(imgs_list)

        for img in imgs_list:
            img_name = img[:-4]
            train_emb_dict[img_name] = [int(track_label)]





    SAVE_PATH = "/ghome/group07/test/W4/part2/triplet_train/pickles/"

    with open(os.path.join(SAVE_PATH, "list_train_emb.pkl"), 'wb') as f:
        pickle.dump(train_emb_list, f)

    with open(os.path.join(SAVE_PATH, "dict_train_emb.pkl"), 'wb') as f:
        pickle.dump(train_emb_dict, f)

    with open(os.path.join(SAVE_PATH, "list_test_emb.pkl"), 'wb') as f:
        pickle.dump(test_emb_list, f)

    with open(os.path.join(SAVE_PATH, "dict_test_emb.pkl"), 'wb') as f:
        pickle.dump(test_emb_dict, f)

        
    
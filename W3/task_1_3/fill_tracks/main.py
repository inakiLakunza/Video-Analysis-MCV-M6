import numpy as np
import pickle


if __name__ == "__main__":
    
    with open('track_updater_pickle.pkl', 'rb') as inp:
        ended_updater = pickle.load(inp)

    ended_updater.fill_missing_tracks()
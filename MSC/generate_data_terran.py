import os
import pickle
import numpy as np
from scipy import sparse
from scipy.io import savemat


dir_path = './parsed_replays/GlobalFeatureVector/Terran_vs_Terran/Terran'

# constants, etc
terran_actions = 75
num_replays =  100
'''
features interested in/indexes:
    Idea 2 state 3 features (army vs worker, idea for assessing skill) 
        Army supply, - 20
        Worker supply - 21
        Idea worker count - 22
'''
feature_indexes = [20, 21, 22] # columns of each replay
feature_signs = [1, 1, -1] # used map features -> rewards


'''
probably 60fps, each entry after 96 frames
so first 4 minutes goes up to about the 150th row
first 3 goes up to about 112.5 row
'''
# each frame is 96 actual frames apart, seems to be 60 frames a second,
# so each replay maybe 1.6 seconds apart, so first 3 minutes of gameplay = 
# 3 minutes * 60 seconds = 180 seconds / 1.6 seconds a replay = 112.5 = 113 frames
num_frames = 113 # number of rows


states = {}
actions = {}
rewards = {}

# iterate through first 100 replays, add data to dictionaries

paths = os.listdir(dir_path)
for i in range(num_replays):
    index = str(i)
    data = np.asarray(sparse.load_npz(os.path.join(dir_path,paths[i])).todense())
    
    # features -> states
    features = data[:num_frames, feature_indexes]
    states[index] = features

    # obtain rewards
    rewards[index] = features * feature_signs

    # append actions
    actions[index] = data[:num_frames, 1]

save_directory = 'exported_replays'



# make a descriptive file name
states_file_name  = 'states_TerranVsTerran_{}_{}_{}'.format(num_replays,num_frames, feature_indexes)
actions_file_name = 'actions_TerranVsTerran_{}_{}_{}'.format(num_replays,num_frames, terran_actions)
rewards_file_name = 'rewards_TerranVsTerran_{}_{}_{}'.format(num_replays,num_frames, feature_signs)

states_file_path = os.path.join(save_directory, states_file_name)
actions_file_path = os.path.join(save_directory, actions_file_name)
rewards_file_path = os.path.join(save_directory, rewards_file_name)


# pickle for python
pickle.dump(states, open(states_file_path + ".pkl", "wb"))
pickle.dump(actions, open(actions_file_path + ".pkl", "wb"))
pickle.dump(rewards, open(rewards_file_path + ".pkl", "wb"))


# save to mat
savemat(states_file_path + '.mat', states)
savemat(actions_file_path + '.mat', actions)
savemat(rewards_file_path + '.mat', rewards)
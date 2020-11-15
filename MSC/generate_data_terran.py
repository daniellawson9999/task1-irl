import os
import pickle
import numpy as np
from scipy import sparse
from scipy.io import savemat
import json

dir_path = './parsed_replays/GlobalFeatureVector/Terran_vs_Terran/Terran'

# constants, etc
terran_actions = 75
num_replays =  100
'''
features interested in/indexes:
    Idea 2 state 3 features (army vs worker, idea for assessing skill) 
        Army supply, - 20
        Worker supply - 21
        Idle worker count - 22
'''
state_indexes = [16,17,18,19,20,21,22,23,24,25,26]
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
#num_frames = 113 # number of rows
num_frames = 150

states = {}
actions = {}
rewards = {}

# iterate through first 100 replays, add data to dictionaries

paths = os.listdir(dir_path)
for i in range(num_replays):
    index = str(i)
    data = np.asarray(sparse.load_npz(os.path.join(dir_path,paths[i])).todense())
    

    # store states
    states[index] = data[:num_frames, state_indexes]

    # obtain rewards
    features = data[:num_frames, feature_indexes]
    rewards[index] = features * feature_signs

    # append actions
    actions[index] = data[:num_frames, 1]


# idea: reduce action space (to build, train, cancel, research, stop, morph, halt
# alt idea: reduce action space to build, train worker, train army, and maybe
macro_action_space = ['Build', 'TrainWorker', 'TrainArmy', 'None']

terran_stat = json.load(open('./parsed_replays/Stat/Terran.json'))
id_reversed = {value: key for (key, value) in terran_stat['action_id'].items()}

# iterate over each player
for i in range(num_replays):
    index = str(i)
    for j in range(num_frames):
        # map actions to macro action space
        action_id = str(int(actions[index][j]))
        if action_id == '74':
            macro_action = 'None'
        else:
            id_r  = id_reversed[str(action_id)]
            # get name of action
            action_name = terran_stat['action_name'][id_r]
            # if SCV action, then map to TrainWorker,
            if action_name == "Train_SCV_quick":
                macro_action = "TrainWorker"
            else:
                macro_action = action_name[:action_name.index('_')]
                # Map to macro action, if a train map to TrainArmy
                if macro_action == "Train":
                    macro_action = "TrainArmy"
                elif macro_action != "Build":
                    macro_action = "None"
        new_action = macro_action_space.index(macro_action)
        actions[index][j] = new_action
    # remove None actions from states, rewards, actions
    valid_states = [False] * num_frames
    for j in range(num_frames):
        # set state to be valid if it does not contain the None action
        valid_states[j] = (actions[index][j] != len(macro_action_space) - 1)
    # filter arrays
    actions[index] = actions[index][valid_states]
    states[index] = states[index][valid_states, :]
    rewards[index] = rewards[index][valid_states, :]

# re-scale rewards to use the standard score = (x - mu) / sigma)
num_rewards = len(rewards['0'][0])
all_rewards = [[] for i in range(num_rewards)] # create arrays for each reward feature
for i in range(num_replays):
    all_rewards = np.concatenate((all_rewards, rewards[str(i)].T), axis = 1)

mus = np.nanmean(all_rewards, axis = 1)
sigmas = np.nanstd(all_rewards, axis = 1)

def standard_score(array, mu, sigma):
    return (array - mu) / sigma

# modify rewards for all players
for i in range(num_replays):
    index = str(i)
    # standardize each column
    for j in range(num_rewards):
        rewards[index][:, j] = standard_score(rewards[index][:, j],
                                              mus[j], sigmas[j])



save_directory = 'exported_replays'

# make a descriptive file name
states_file_name  = 'states_TerranVsTerran_{}_{}_[{}:{}]'.format(num_replays,num_frames, state_indexes[0], state_indexes[-1])
actions_file_name = 'actions_TerranVsTerran_{}_{}_{}'.format(num_replays,num_frames, len(macro_action_space) - 1)
rewards_file_name = 'rewards_TerranVsTerran_{}_{}_{}'.format(num_replays,num_frames, np.array(feature_indexes) * feature_signs)

states_file_path = os.path.join(save_directory, states_file_name)
actions_file_path = os.path.join(save_directory, actions_file_name)
rewards_file_path = os.path.join(save_directory, rewards_file_name)

# pickle for python
pickle.dump(states, open(states_file_path + ".pkl", "wb"))
pickle.dump(actions, open(actions_file_path + ".pkl", "wb"))
pickle.dump(rewards, open(rewards_file_path + ".pkl", "wb"))
# rename keys for MATLAB
for i in range(len(states.keys())):
    key = list(states.keys())[0]
    states["id" + key] = states.pop(key)
    actions["id" + key] = actions.pop(key)
    rewards["id" + key] = rewards.pop(key)

# save to mat
savemat(states_file_path + '.mat', states)
savemat(actions_file_path + '.mat', actions)
savemat(rewards_file_path + '.mat', rewards)


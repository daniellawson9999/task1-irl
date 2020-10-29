import os
import numpy as np
from scipy import sparse
from scipy.io import savemat


dir_path = './parsed_replays/GlobalFeatureVector/Terran_vs_Terran/Terran'

# contants
num_replays =  100
feature_indexes = [20, 21, 22] # columns of each replay
# each frame is 96 actual frames apart, seems to be 60 frames a second,
# so each replay maybe 1.6 seconds apart, so first 3 minutes of gameplay = 
# 3 minutes * 60 seconds = 180 seconds / 1.6 seconds a replay = 112.5 = 113 frames
num_frames = 113 # number of rows

'''
features interested in/indexes:
    Idea 2 state 3 features (army vs worker, idea for assessing skill) 
        Army supply, - 20
        Worker supply - 21
        Idea worker count - 22
'''

'''
probably 60fps, each entry after 96 frames
so first 4 minutes goes up to about the 150th row
first 3 goes up to about 112.5 row
'''


# iterate through first 100 replays, appends features to data array
data = []
paths = os.listdir(dir_path)
for i in range(num_replays):
    features = np.asarray(sparse.load_npz(os.path.join(dir_path,paths[i])).todense())
    # reshape to rows and columns we want
    features = features[:num_frames, feature_indexes]
    # add id column
    features = np.insert(features, 0, i, axis = 1)
    data.append(features)

# convert to 3D numpy array 
data = np.array(data)

save_directory = 'exported_replays'
# make a descriptive file name
file_name  =  'TerranVsTerran_{}_{}_{}'.format(num_replays,num_frames, feature_indexes)

file_path = os.path.join(save_directory, file_name)

# save to numpy
np.save(file_path, data)

# save to mat
savemat(file_path + '.mat', {"data": data})
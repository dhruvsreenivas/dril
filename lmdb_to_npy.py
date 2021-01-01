import numpy as np
import os.path as osp
import lmdb
import pyarrow as pa
from copy import deepcopy

def lmdb_to_npy(lmdb_file_path):
    db = lmdb.open(lmdb_file_path, subdir=osp.isdir(lmdb_file_path), readonly=True, lock=False, readahead=False, meminit=False)
    with db.begin(write=False) as txn:
        keys = pa.deserialize(txn.get(b'__keys__'))
        length = pa.deserialize(txn.get(b'__len__'))
    
    state_list = []
    action_list = []

    for idx in range(length):
        if keys is not None:
            with db.begin(write=False) as txn:
                byteflow = txn.get(keys[idx])
            state, action, _ = pa.deserialize(byteflow)
            state, action = deepcopy(state), deepcopy(action)
        else:
            state, action, _ = db[idx]


        state_list.append(state)
        action_list.append(action)

    all_states = np.array(state_list)
    all_actions = np.array(action_list)
    np.save('./demo_data_pong/obs_PongNoFrameskip-v4_seed=69_ntraj=20.npy', all_states)
    np.save('./demo_data_pong/acs_PongNoFrameskip-v4_seed=69_ntraj=20.npy', all_actions)

if __name__ == '__main__':
    lmdb_to_npy('./pong_expert_trajs/expert_samples_20_sticky.lmdb')

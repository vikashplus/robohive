import pickle
import os
from robohive.logger.reference_motion import ReferenceMotion

curr_dir = os.path.dirname(os.path.abspath(__file__))


data_path = '/Subj05jumpIK.pkl'

data = pickle.load(open(curr_dir+data_path, 'rb'))

# data['robot_vel'] = None
# data['object'] = None
# data['robot_vel'] = None

ref = ReferenceMotion(data)

import ipdb; ipdb.set_trace()

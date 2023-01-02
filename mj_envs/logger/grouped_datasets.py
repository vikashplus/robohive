from mj_envs.utils import tensor_utils
from mj_envs.utils.dict_utils import flatten_dict, dict_numpify
import numpy as np
import pickle
import h5py
from PIL import Image
from sys import platform
import skvideo.io
import os

# Trace_name: {
#     grp1: {dataset{k1:v1}, dataset{k2:v2}, ...}
#     grp2: {dataset{kx:vx}, dataset{ky:vy}, ...}
# }

# ToDo
# access pattern for pickle and h5 backbone post load isn't the same
# Should we get rid of pickle support and double down on h5?
class Trace:
    def __init__(self, name):
        self.name = name
        self.root = {name: {}}
        self.trace = self.root[name]


    # Create a group in your logs
    def create_group(self, name):
        self.trace[name] = {}


    # Directly add a full dataset to a given group
    def create_dataset(self, group_key, dataset_key, dataset_val):
        if group_key not in self.trace.keys():
            self.create_group(name=group_key)
        self.trace[group_key][dataset_key] = [dataset_val]


    # Remove dataset from an existing group(s)
    def remove_dataset(self, group_keys:list, dataset_key:str):
        if type(group_keys)==str:
            if group_keys==":":
                group_keys = self.trace.keys()
            else:
                group_keys=[group_keys]

        for group_key in group_keys:
            assert group_key in self.trace.keys(), "Group:{} does not exist".format(group_key)
            if dataset_key in self.trace[group_key].keys():
                del self.trace[group_key][dataset_key]


    # Append dataset datum to an existing group
    def append_datum(self, group_key, dataset_key, dataset_val):
        assert group_key in self.trace.keys(), "Group:{} does not exist".format(group_key)
        if dataset_key in self.trace[group_key].keys():
            self.verify_type(dataset=self.trace[group_key][dataset_key], data=dataset_val)
            self.trace[group_key][dataset_key].append(dataset_val)
        else:
            self.trace[group_key][dataset_key] = [dataset_val]


    # Append dataset dict{datums} to an existing group
    def append_datums(self, group_key:str, dataset_key_val:dict)->None:
        for dataset_key, dataset_val in dataset_key_val.items():
            self.append_datum(group_key=group_key, dataset_key=dataset_key, dataset_val=dataset_val)


    # Get data
    def get(self, group_key, dataset_key, dataset_ind=None):
        if dataset_ind is None:
            return self.trace[group_key][dataset_key]
        else:
            return self.trace[group_key][dataset_key][dataset_ind]


    # Set data
    def set(self, group_key, dataset_key, dataset_ind=None, dataset_val=None):
        if dataset_ind is None:
            self.trace[group_key][dataset_key] = [dataset_val]
        else:
            self.verify_type(dataset=self.trace[group_key][dataset_key], data=dataset_val)
            self.trace[group_key][dataset_key][dataset_ind] = dataset_val


    # verify if a data can be a part of an existing datasets
    def verify_type(self, dataset, data):
        dataset_type = type(dataset[0])
        assert type(data) == dataset_type, TypeError("Type mismatch while appending. Datum should be {}".format(dataset_type))


    # Verify that all datasets in each groups are of same length. Helpful for time synced traces
    def verify_len(self):
        for grp_k, grp_v in self.trace.items():
            dataset_keys = grp_v.keys()
            for i_key, key in enumerate(dataset_keys):
                if i_key == 0:
                    trace_len = len(self.trace[grp_k][key])
                else:
                    key_len = len(self.trace[grp_k][key])
                    assert trace_len == key_len, ValueError("len({}[{}]={}, should be {}".format(grp_k, key, key_len, trace_len))


    # Very if trace is stacked and flattened. Useful for utilities like render, save etc
    def verify_stacked_flattened(self):
        for grp_k, grp_v in self.trace.items():
            for dst_k, dst_v in grp_v.items():
                # Check if stacked
                if type(dst_v) == list:
                    return False
                # check if flattened
                if type(dst_v) == dict:
                    return False
        return True


    # Render frames/videos
    def render(self, output_dir, output_format, groups:list, datasets:list, input_fps:int=25):
        # output_dir:       path for output
        # output_type:      rgb/ mp4
        # groups:           Groups to render: Pass ":" for rendering given dataset from all groups
        # datasets:         List(datasets) to render Example ['left', 'right', 'top', 'Franka_wrist']
        #                   dataset can be np.ndarray([N,H,W,3])stacked or a list Nx[HxWx3]
        # input_fps         input fps of the provided dataset frames

        # Resolve groups
        if type(groups)==str:
            if groups==":":
                groups = self.trace.keys()
            else:
                groups = [groups]
        for grp in groups:
            assert grp in self.trace.keys(), "Unknown group {}".format(grp)

        # Run through all trajs in the paths
        for i_grp, grp in enumerate(groups):

            # Pre allocate buffer
            if type(self.trace[grp][datasets[0]])==list: #unstacked
                horizon = len(self.trace[grp][datasets[0]])
                height, width, _ = self.trace[grp][datasets[0]][0].shape
            elif type(self.trace[grp][datasets[0]])==np.ndarray: #stacked
                horizon, height, width, _ = self.trace[grp][datasets[0]].shape

            frame_tile = np.zeros((height, width*len(datasets), 3), dtype=np.uint8)
            if output_format == "mp4":
                frames = np.zeros((horizon, height, width*len(datasets), 3), dtype=np.uint8)

            # Render
            print("Recovering {} frames:".format(output_format), end="")
            for t in range(horizon):
                # render single frame
                for i_cam, cam_key in enumerate(datasets):
                    frame_tile[:,i_cam*width:(i_cam+1)*width, :] = self.trace[grp][cam_key][t]
                # process single frame
                if output_format == "mp4":
                    frames[t,:,:,:] = frame_tile
                elif output_format == "rgb":
                    image = Image.fromarray(frame_tile)
                    file_name_rgb = os.path.join(output_dir, grp+'-'+str(t)+".png")
                    image.save(file_name_rgb)
                else:
                    raise TypeError("Unknown format")
                print(t, end=",", flush=True)

            # Save video
            if output_format == "mp4":
                file_name_mp4 = os.path.join(output_dir, grp+".mp4")
                inputdict={"-r": str(input_fps)}
                # quicktime compatibility for mac-os
                if platform == "darwin":
                    skvideo.io.vwrite(file_name_mp4, np.asarray(frames),inputdict=inputdict, outputdict={"-pix_fmt": "yuv420p"})
                else:
                    skvideo.io.vwrite(file_name_mp4, np.asarray(frames),inputdict=inputdict)
                print("\nSaved: " + file_name_mp4)


    # Display data
    def __repr__(self) -> str:
        disp = "Trace_name: {}\n".format(self.root.keys())
        for grp_k, grp_v in self.trace.items():
            disp += "{"+grp_k+": \n"
            for dst_k, dst_v in grp_v.items():

                # raw
                if type(dst_v) == list:
                    datum = dst_v[0]
                    try:
                        ll = datum.shape
                    except:
                        ll = ()
                    disp += "\t{}:[{}_{}]_{}\n".format(dst_k, str(type(dst_v[0])), ll, len(dst_v))

                # flattened
                elif type(dst_v) == dict:
                    datum = dst_v
                    disp += "\t{}: {}\n".format(dst_k, str(type(datum)))

                # numpified
                else:
                    datum = dst_v
                    disp += "\t{}: {}, shape{}, type({})\n".format(dst_k, str(type(datum)), datum.shape, datum.dtype)

            disp += "}\n"
        return disp


    # Stack trace
    def stack(self):
        for grp_k, grp_v in self.trace.items():
            for dst_k, dst_v in grp_v.items():
                if type(dst_v)==list and type(dst_v[0]) == dict:
                    grp_v[dst_k] = tensor_utils.stack_tensor_dict_list(dst_v)
                elif type(dst_v)==list and type(dst_v[0]) != str:
                    grp_v[dst_k] = np.array(dst_v)


    # Flatten
    def flatten(self):
        for grp_k, grp_v in self.trace.items():
            self.trace[grp_k] = flatten_dict(data=grp_v)


    # Numpify everything
    def numpify(self, u_res, i_res, f_res):
        for grp_k, grp_v in self.trace.items():
            self.trace[grp_k] = dict_numpify(data=grp_v, u_res=u_res, i_res=i_res, f_res=f_res)


    # Close the logger and post process the data
    def close(self,
            u_res=np.uint8, i_res=np.int8, f_res=np.float16,
            verify_length=False):

        # stack all records
        self.stack()

        # flatten structure
        self.flatten() # WARNING: Will create loading difference between h5 and pickle backbones

        # fix datatypes and resolutions
        self.numpify(u_res=u_res, i_res=i_res, f_res=f_res)

        # verify that
        if verify_length:
            self.verify_len()


    # Save
    def save(self,
                # save options
                trace_name:str,
                # compression options
                compressions='gzip',
                compression_opts=4,
                **kwargs
                ):

        # close trace before saving
        if not self.verify_stacked_flattened():
            print("Closing Trace: "+self.name)
            self.close(**kwargs)

        # save
        trace_format = trace_name.split('.')[-1]
        if trace_format == "h5":
            paths_h5 = h5py.File(trace_name, "w")
            for grp_k, grp_v in self.trace.items():
                trial = paths_h5.create_group(grp_k)
                for dst_k, dst_v in grp_v.items():
                    trial.create_dataset(dst_k, data=dst_v, compression='gzip', compression_opts=compression_opts)
        else:
            pickle.dump(self.root, open(trace_name, 'wb'))
        print("Saved: "+trace_name)


    # load trace from disk
    def load(trace_path):
        trace_format = trace_path.split('.')[-1]
        print("Reading: ", trace_path)
        if trace_format == "h5":
            trace = h5py.File(trace_path, "r")
        else:
            trace = pickle.load(open(trace_path, 'rb'))
        return trace


# Test trace
def test_trace():
    trace = Trace("Root_name")

    # Create a group: append and verify
    trace.create_group("grp1")
    trace.create_dataset(group_key="grp1", dataset_key="dst_k1", dataset_val="dst_v1")
    trace.append_datum(group_key="grp1", dataset_key="dst_k1", dataset_val="dst_v11")
    trace.create_dataset(group_key="grp1", dataset_key="dst_k2", dataset_val="dst_v2")
    trace.append_datum(group_key="grp1", dataset_key="dst_k2", dataset_val="dst_v22")
    trace.verify_len()

    # Add another group
    trace.create_group("grp2")
    trace.create_dataset(group_key="grp2", dataset_key="dst_k3", dataset_val={"dst_v3":[3]})
    trace.create_dataset(group_key="grp2", dataset_key="dst_k4", dataset_val={"dst_v4":[4]})
    print(trace)

    # get set methods
    datum = "dst_v111"
    trace.set('grp1','dst_k1', 0, datum)
    assert datum == trace.get('grp1','dst_k1', 0), "Get-Set error"
    datum = {"dst_v33":[33]}
    trace.set('grp2','dst_k4', 0, datum)
    assert datum == trace.get('grp2','dst_k4', 0), "Get-Set error"

    # save-load methods
    trace.save(trace_name='test_trace.pickle', verify_length=True)
    trace.save(trace_name='test_trace.h5', verify_length=True)

    h5_trace = Trace.load("test_trace.h5")
    pkl_trace = Trace.load("test_trace.pickle")

    print("H5 trace")
    print(h5_trace)
    print("PKL trace")
    print(pkl_trace)

if __name__ == '__main__':
    test_trace()






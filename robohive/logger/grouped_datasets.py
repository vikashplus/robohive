from robohive.utils import tensor_utils
from robohive.utils.dict_utils import flatten_dict, dict_numpify
from robohive.utils.prompt_utils import prompt, Prompt
import numpy as np
import pickle
import h5py
from PIL import Image
from sys import platform
import skvideo.io
import os
import enum

# Trace_name: {
#     grp1: {dataset{k1:v1}, dataset{k2:v2}, ...}
#     grp2: {dataset{kx:vx}, dataset{ky:vy}, ...}
# }

# ToDo
# access pattern for pickle and h5 backbone post load isn't the same
#   - Should we get rid of pickle support and double down on h5?
#   - other way would to make the default container (trace.trace) h5 container instead of a dict


class TraceType(enum.Enum):
    """Trace types."""
    UNSET = -1
    ROBOHIVE = 0
    ROBOSET = 1

    def get_type(input_type):
        """
        A more robust way of getting trace type. Supports strings
        """
        if isinstance(input_type, str):
            if input_type.lower() == "robohive":
                return TraceType.ROBOHIVE
            elif input_type.lower() == "roboset":
                return TraceType.ROBOSET
            else:
                prompt(f"unknown TraceType{input_type}. Setting it to TraceType.UNSET", type=Prompt.WARN)
                return TraceType.UNSET


class Trace:
    def __init__(self, name):
        self.name = name
        self.root = {name: {}}
        self.trace = self.root[name]
        self.index = 0
        self.type = TraceType.ROBOHIVE
        self.closed = False     # False: Trace is open for edits. True: Trace can be analyzed but not edited.

    # Create a group in your logs
    def create_group(self, name):
        self.trace[name] = {}


    # Directly add a full dataset to a given group. If data appending is needed, use create_datum() instead
    def create_dataset(self, group_key, dataset_key, dataset_val):
        if group_key not in self.trace.keys():
            self.create_group(name=group_key)
        self.trace[group_key][dataset_key] = dataset_val


    # Remove dataset from an existing group(s)
    def remove_dataset(self, group_keys:list, dataset_key:str):
        if isinstance(group_keys, str):
            if group_keys==":":
                group_keys = self.trace.keys()
            else:
                group_keys=[group_keys]

        for group_key in group_keys:
            assert group_key in self.trace.keys(), "Group:{} does not exist".format(group_key)
            if dataset_key in self.trace[group_key].keys():
                del self.trace[group_key][dataset_key]


    # Create the first datum of an existing group. Use append_datum() to append more elements
    def create_datum(self, group_key, dataset_key, dataset_val):
        if group_key not in self.trace.keys():
            self.create_group(name=group_key)
        self.trace[group_key][dataset_key] = [dataset_val]


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
    def get(self, group_key, dataset_key=None, dataset_ind=None):
        if dataset_ind is None:
            return self.trace[group_key]
        elif dataset_ind is None:
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
        assert isinstance(data, dataset_type), TypeError("Type mismatch while appending. Datum should be {}".format(dataset_type))

        # check for array
        if isinstance(data, np.ndarray):
            assert data.shape == dataset[0].shape, ValueError(f"Data dimenstion({data.shape}) not compatible with dataset dimensions({dataset[0].shape})")
        # check for list
        if isinstance(data, list):
            assert len(data) == len(dataset[0]), ValueError(f"Data dimenstion({len(data)}) not compatible with dataset dimensions({len(dataset[0])})")
        # check for dictionary
        if isinstance(data, dict):
            flattened_data = flatten_dict(data)
            flattened_dataset = flatten_dict(dataset[0])
            assert flattened_data.keys() == flattened_dataset.keys(), ValueError(f"Data keys {flattened_data.keys()} not compatible with dataset keys {flattened_dataset.keys()}")
            for key in flattened_data:
                assert np.array(flattened_data[key]).shape == np.array(flattened_dataset[key]).shape, ValueError(f"Data dimension for key '{key}' ({np.array(flattened_data[key]).shape}) not compatible with dataset dimensions ({np.array(flattened_dataset[key]).shape})")


    # Verify that all datasets in each groups are of same length. Helpful for time synced traces
    def verify_len(self):
        for grp_k, grp_v in self.trace.items():
            dataset_keys = grp_v.keys()
            for i_key, key in enumerate(dataset_keys):
                if i_key == 0:
                    trace_len = len(self.trace[grp_k][key])
                else:
                    key_len = len(self.trace[grp_k][key])
                    assert trace_len == key_len, ValueError("Dataset length mismatch: len({}[{}]={}, should be {}".format(grp_k, key, key_len, trace_len))


    # Very if trace is stacked and flattened. Useful for utilities like render, save etc
    def verify_stacked_flattened(self):
        if self.closed:
            True

        for grp_k, grp_v in self.trace.items():
            for dst_k, dst_v in grp_v.items():
                # Check if stacked
                if type(dst_v) == list:
                    return False
                # check if flattened
                if type(dst_v) == dict:
                    return False
        return True

    # plot data
    def plot(self, output_dir, output_format, groups:list, datasets:list, x_dataset:str='time'):
        # Plot dataset traces using the groups and datasets keys list. T
        # ARGUMENTS:
        #   output_dir:       path for output
        #   output_format:    pdf/png/None(for onscreen)
        #   groups:           - list(Groups)_ng to plot:
        #                     - ng = len(groups) == number of subplots
        #                     - ":" to consider each group once
        #                     - None entry in the list can will leave the subplot empty
        #   datasets:         - list(list(datasets))_ng to plot([['left',], ['right', 'top']]),
        #                     - ng = len(groups) == len(datasets) == number of subplots
        #                     - ":" to plot each dataset once
        #   x_dataset:        - dataset key to use as x-axis if available
        # EXAMPLES
        #   1. plot(..., groups=":", data=":")
        #       produces a plot with len(groups) subplots
        #   2. plot(...,groups=['traj1', 'traj1', 'traj2'], data=[['qpos'], ['qpos','qvel'], ['qvel']])
        #       produces a plot with three subplots

        if not self.closed:
            prompt("Trace is still open for edits. Close the trace to enable plotting", type=Prompt.WARN)
            return

        import matplotlib as mpl
        # mpl.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 5})
        plt.clf()

        # Resolve groups
        if isinstance(groups, str) and groups==":":
            groups = list(self.trace.keys())
        elif isinstance(groups, str):
            groups = [groups]
        else:
            assert isinstance(groups, list), TypeError(f"Expected a list of groups. Got {groups}")

        # number of subplots
        n_subplot = len(groups)

        # Check for datasets
        if isinstance(datasets, str) and datasets==":":
            datasets = n_subplot*[":"]
        elif isinstance(datasets, str):
            datasets = [datasets]
        else:
            assert (isinstance(datasets, list)), TypeError(f"Dataset keys needs to be a list. Got {datasets}")

        # Check for group and datasets sizes
        assert len(datasets)==n_subplot, ValueError(f"len(groups):{n_subplot} has to match len(datasets):{len(datasets)}")
        # print(groups)
        # print(datasets)

        # Run through all groups
        for i_grp, grp_key in enumerate(groups):

            # Leave empty if requested
            if grp_key is None:
                continue

            # process group / subplot
            assert isinstance(grp_key, str), TypeError(f"Dataset key needs to be a string. Got {grp_key}")
            assert grp_key in self.trace.keys(), "Unknown group {}. Available groups {}".format(grp_key, self.trace.keys())
            grp_val = self.trace[grp_key]

            # print('selected group', grp_key)

            # Resolve datasets within existing group
            if isinstance(datasets, str) and datasets==":":
                i_grp_datasets = list(grp_val.keys())
            elif isinstance(datasets[i_grp], str) and datasets[i_grp]==":":
                i_grp_datasets = list(grp_val.keys())
            else:
                i_grp_datasets = datasets[i_grp]

            assert isinstance(i_grp_datasets, list) and isinstance(i_grp_datasets[0], str), TypeError(f"Unrecognized dataset input for group:{grp_key}. Expected ':', or a list from {grp_val.keys()}. Got: {i_grp_datasets}")

            # Run through all dataset requests within the group
            for ds_key in i_grp_datasets:
                assert ds_key in grp_val.keys(), f"Group: {grp_key} :> Unknown dataset {ds_key}. Available datasets {grp_val.keys()}"
                ds_val = grp_val[ds_key]

                assert isinstance(ds_val, np.ndarray), ValueError(f"Dataset for plotting needs to be an array. Provided data:{ds_val}, type:{type(ds_val)}")
                assert np.issubdtype(ds_val.dtype, np.number), ValueError(f"Dataset for plotting needs to of numerical dtype. Provided dtype: {ds_val.dtype}")
                assert len(ds_val.shape)<3, ValueError(f"Plotting is only supported for 1D and 2D Dataset. Provided data dims: {ds_val.shape}")

                # print(f"g:{grp_key}/ d:{ds_key}")
                h_axis = plt.subplot(n_subplot, 1, i_grp+1)
                # h_axis.set_prop_cycle(None)

                if x_dataset in grp_val.keys():
                    plt.plot(grp_val[x_dataset], ds_val, label=f"{grp_key}/{ds_key}", marker='*')
                    h_axis.set_xlabel(x_dataset)
                else:
                    plt.plot(ds_val, label=f"{grp_key}/{ds_key}", marker='*')
                h_axis.set_title(grp_key)
                h_axis.legend()


        # show/save plot
        if output_format is None:
            plt.show()
        else:
            file_name = os.path.join(output_dir, f"{self.name}_{grp_key}_{ds_key}_{output_format}".replace("/", "_"))
            plt.savefig(file_name)
            print("saved ", file_name)


    # Render frames/videos
    def render(self, output_dir, output_format, groups:list, datasets:list, input_fps:int=25):
        # output_dir:       path for output
        # output_format:    rgb/ mp4
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
            assert grp in self.trace.keys(), "Unknown group {}. Available groups {}".format(grp, self.trace.keys())

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


    def __getitem__(self, index):
        """
            Enables indexing using either index(int) or keys(Trial0)
            Example: Data = Trace(); Data[0] == Data['Trial0']
        """
        if type(index) == str:
            assert index in self.trace.keys(), f"Index({index}) not in existing keys({list(self.trace.keys())})"
            return self.trace[index]
        elif type(index) == int:
            assert index<len(self), f"Index({index}) outside the max lenght({len(self)})"
            keys = list(self.trace.keys())
            key = keys[index]
            value = self.trace[key]
            return value
        else:
            raise TypeError(f"index has to be str(TrailX), or int. {index} found")


    def __iter__(self):
        """
        Enables iteration over trace's groups. Makes it look like a list of groups
        """
        return self


    def __next__(self):
        """
        Enables iteration over trace's groups. Makes it look like a list of groups
        """
        if self.index >= len(self):
            self.index = 0
            raise StopIteration

        item = self[self.index]
        # keys = list(self.trace.keys())
        # value = self.trace[keys[self.index]]
        self.index += 1
        return item

    def items(self):
        """
        Enables iteration over trace with key-value pairs
        """
        return zip(self.trace.keys(), self)

    # return length
    def __len__(self) -> str:
        """
        returns the number of groups in the trace
        """
        return len(self.trace.keys())


    # Display data
    def __repr__(self) -> str:
        disp = "Trace_name: {}\n".format(self.root.keys())

        if isinstance(self.trace, h5py.File):
        # Trace (when reloaded from h5)
            for k, v in self.trace.items():
                disp += v.__repr__()+"\n"
                for kk,vv in v.items():
                    disp += "\t"+vv.__repr__()+"\n"

        else:
        # Trace (while open)
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
        """
        Close the logs by stacking, flattening, and numpyfies. This an irreversible change
        """

        # stack all records
        self.stack()

        # flatten structure
        self.flatten() # WARNING: Will create loading difference between h5 and pickle backbones

        # fix datatypes and resolutions
        self.numpify(u_res=u_res, i_res=i_res, f_res=f_res)

        # verify that
        if verify_length:
            self.verify_len()

        self.closed = True


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
                    trial.create_dataset(dst_k, data=dst_v, compression=compressions, compression_opts=compression_opts)
        else:
            pickle.dump(self.root, open(trace_name, 'wb'))
        print("Saved: "+trace_name)


    # load trace from disk
    @staticmethod
    def load(trace_path, trace_type=TraceType.UNSET):
        """
        trace_path: Load the trace using the provided path
        trace_type: Provide the trace type of the path; UNSET will be used if not provided
        Note:
            Loaded trace has some difference with the original trace
            - h5 vs dict format
            - flattend schema
        """
        trace_name, trace_format = os.path.splitext(trace_path)
        print("Reading:", trace_path)
        if trace_format == ".h5":
            trace = Trace(name=trace_name)
            trace.trace_type=TraceType.get_type(trace_type)
            file_data = h5py.File(trace_path, "r")
            trace.trace = file_data # load data
            trace.root[trace.name] = trace.trace # build root
        else:
            file_data = pickle.load(open(trace_path, 'rb'))
            trace = Trace(name=list(file_data.keys())[0])
            trace.trace = file_data[trace.name] # load data
            trace.root = file_data  # build root
            trace.trace_type=TraceType.get_type(trace_type)
        return trace


def test_trace_plot():
    trace = Trace("root_name")

    data1 = np.sin(np.arange(0,100))
    data2 = np.cos(np.arange(0,100))
    data3 = np.sin(np.arange(0,200))+np.cos(np.arange(0,200))
    time = 0.01*np.arange(0,200)

    trace.create_group("grp1")
    trace.create_dataset(group_key="grp1", dataset_key="dst1", dataset_val=data1)
    trace.create_dataset(group_key="grp1", dataset_key="dst2", dataset_val=data2)

    trace.create_group("grp2")
    trace.create_dataset(group_key="grp2", dataset_key="time", dataset_val=time)
    trace.create_dataset(group_key="grp2", dataset_key="dst3", dataset_val=data3)
    trace.close()

    trace.plot(output_format='plot0.pdf', output_dir=".", groups=["grp1",], datasets=[["dst1",],], x_dataset="dst1")
    trace.plot(output_format='plot1.pdf', output_dir=".", groups=":", datasets=":")
    trace.plot(output_format='plot2.pdf', output_dir=".", groups=":", datasets=[":", ":"])
    trace.plot(output_format='plot3.pdf', output_dir=".", groups=":", datasets=[["dst2",], ":"])

    # Catch issues plotting string array
    try:
        trace = Trace("string")
        trace.create_dataset(group_key="grp1", dataset_key="dst_k1", dataset_val=np.array(["v1", "v2", "v3"]))
        trace.close()
        trace.plot(output_format=None, output_dir=".", groups=["grp1",], datasets=[["dst_k1",],])
    except Exception as e:
        prompt(f"EXPECTED: Caught exception while trying to plot array of strings: {e}", type=Prompt.WARN)

    # Catch issues plotting list(strings)
    try:
        trace = Trace("string")
        trace.create_dataset(group_key="grp1", dataset_key="dst_k1", dataset_val=["v1", "v2", "v3"])
        trace.close()
        trace.plot(output_format='plot4.pdf', output_dir=".", groups=["grp1",], datasets=[["dst_k1",],])
    except Exception as e:
        prompt(f"EXPECTED: Caught exception while trying to plot list of strings: {e}", type=Prompt.WARN)


    # plot complex dicts
    trace = Trace("root_dict")
    trace.create_datum(group_key="grp1", dataset_key="dst_k1", dataset_val={"one":1, "two":2.0, "three":"3"})
    trace.append_datum(group_key="grp1", dataset_key="dst_k1", dataset_val={"one":11, "two":22.0, "three":"33"})
    trace.append_datum(group_key="grp1", dataset_key="dst_k1", dataset_val={"one":111, "two":222.0, "three":"333"})
    trace.close()
    trace.plot(output_format='plot5.pdf', output_dir=".", groups=["grp1"], datasets=[["dst_k1/one","dst_k1/two"],])
    trace.plot(output_format='plot6.pdf', output_dir=".", groups=["grp1","grp1"], datasets=[["dst_k1/one"],["dst_k1/two"]])
    trace.plot(output_format='plot7.pdf', output_dir=".", groups=["grp1","grp1"], datasets=[["dst_k1/one"],["dst_k1/one","dst_k1/two",]])
    trace.plot(output_format='plot8.pdf', output_dir=".", groups=[None, "grp1"], datasets=[None, ["dst_k1/one","dst_k1/two"]])
    # catch trying to plot strings
    try:
        trace.plot(output_format=None, output_dir=".", groups=["grp1"], datasets=[":"])
    except Exception as e:
        prompt(f"EXPECTED: Caught exception while trying to plot dict with strings: {e}", type=Prompt.WARN)


    # Catch trying to plot >2D array
    trace = Trace("root_3darray")
    trace.create_group("grp1")
    trace.create_dataset(group_key="grp1", dataset_key="dst_k1", dataset_val=np.ones([4, 2, 4]))
    trace.close()
    try:
        trace.plot(output_format='plot9.pdf', output_dir=".", groups=["grp1"], datasets=[["dst_k1",],])
    except Exception as e:
        prompt(f"EXPECTED: Caught expected exception during plotting >2D dataset: {e}", type=Prompt.WARN)


# Test trace
def test_trace():
    trace = Trace("Root_name")

    # Create a group: append and verify
    trace.create_group("grp1")
    trace.create_datum(group_key="grp1", dataset_key="dst_k1", dataset_val="dst_v1")
    trace.append_datum(group_key="grp1", dataset_key="dst_k1", dataset_val="dst_v11")
    trace.create_datum(group_key="grp1", dataset_key="dst_k2", dataset_val="dst_v2")
    trace.append_datum(group_key="grp1", dataset_key="dst_k2", dataset_val="dst_v22")
    trace.verify_len()

    # Add another group
    trace.create_group("grp2")
    trace.create_datum(group_key="grp2", dataset_key="dst_k3", dataset_val={"dst_v3":[3]})
    trace.create_datum(group_key="grp2", dataset_key="dst_k4", dataset_val={"dst_v4":[4]})
    print(trace)

    # get set methods
    datum = "dst_v111"
    trace.set('grp1','dst_k1', 0, datum)
    assert datum == trace.get('grp1','dst_k1', 0), "Get-Set error"
    datum = {"dst_v4":[0]}
    trace.set('grp2','dst_k4', 0, datum)
    assert datum == trace.get('grp2','dst_k4', 0), "Get-Set error"
    try:
        datum = {"dst_diff_name":[33]}
        trace.set('grp2','dst_k4', 0, datum)
    except Exception as e:
            prompt(f"Caught expected exception trying to insert an inconsistent datum: {e}", type=Prompt.WARN)

    # save-load methods
    trace.save(trace_name='test_trace.pickle', verify_length=True)
    trace.save(trace_name='test_trace.h5', verify_length=True)

    h5_trace = Trace.load("test_trace.h5")
    pkl_trace = Trace.load("test_trace.pickle")

    print("H5 trace")
    print(h5_trace)
    print("PKL trace")
    print(pkl_trace)


def test_trace_append():
    # Create a group: append str
    trace = Trace("string")
    trace.create_group("grp1")
    trace.create_datum(group_key="grp1", dataset_key="dst_k1", dataset_val="dst_v1")
    trace.append_datum(group_key="grp1", dataset_key="dst_k1", dataset_val="dst_v11")
    trace.append_datum(group_key="grp1", dataset_key="dst_k1", dataset_val="dst_v111")
    # print(trace)
    trace.close()
    print(trace)

    # Create a group: append list(string)
    trace = Trace("list(string)")
    trace.create_group("grp1")
    trace.create_datum(group_key="grp1", dataset_key="dst_k1", dataset_val=["dst_v1","dst_v2"])
    trace.append_datum(group_key="grp1", dataset_key="dst_k1", dataset_val=["dst_v11","dst_v22"])
    trace.append_datum(group_key="grp1", dataset_key="dst_k1", dataset_val=["dst_v111","dst_v222"])
    # print(trace)
    trace.close()
    print(trace)

    # Create a group: append list(float)
    trace = Trace("list(float)")
    trace.create_group("grp1")
    trace.create_datum(group_key="grp1", dataset_key="dst_k1", dataset_val=[1, 2])
    trace.append_datum(group_key="grp1", dataset_key="dst_k1", dataset_val=[11, 22])
    trace.append_datum(group_key="grp1", dataset_key="dst_k1", dataset_val=[111, 222])
    # print(trace)
    trace.close(i_res=np.int16)
    print(trace)

    # Create a group: append dict
    trace = Trace("dict")
    trace.create_group("grp1")
    trace.create_datum(group_key="grp1", dataset_key="dst_k1", dataset_val={"one":1, "two":2.0, "three":"3"})
    trace.append_datum(group_key="grp1", dataset_key="dst_k1", dataset_val={"one":11, "two":22.0, "three":"33"})
    trace.append_datum(group_key="grp1", dataset_key="dst_k1", dataset_val={"one":111, "two":222.0, "three":"333"})
    # print(trace)
    trace.close()
    print(trace)

    # Create a group: append ndarray
    trace = Trace("ndarray")
    trace.create_group("grp1")
    trace.create_datum(group_key="grp1", dataset_key="dst_k1", dataset_val=np.array([1, 2]))
    trace.append_datum(group_key="grp1", dataset_key="dst_k1", dataset_val=np.array([11, 22]))
    trace.append_datum(group_key="grp1", dataset_key="dst_k1", dataset_val=np.array([111, 222]))
    print(trace)
    trace.close(i_res=np.int16)
    print(trace)

    # Create a group: append ndarray
    trace = Trace("ndarray_stack")
    trace.create_group("grp1")
    trace.create_datum(group_key="grp1", dataset_key="dst_k1", dataset_val=np.ones([4, 2]))
    trace.append_datum(group_key="grp1", dataset_key="dst_k1", dataset_val=np.zeros([4, 2]))
    try:
        trace.append_datum(group_key="grp1", dataset_key="dst_k1", dataset_val=np.array([11, 22]))
    except Exception as e:
        prompt(f"Caught expected exception during append_datum: {e}", type=Prompt.WARN)
    trace.close(i_res=np.int16)
    assert trace['grp1']['dst_k1'].shape==(2, 4 ,2), ValueError("Check ndarray concatenation")
    try:
        trace.plot(output_format=None, output_dir=".", groups=["grp1"], datasets=[["dst_k1",],])
    except Exception as e:
        prompt(f"Caught expected exception during plotting >2D dataset: {e}", type=Prompt.WARN)



if __name__ == '__main__':
    test_trace()
    test_trace_append()
    test_trace_plot()

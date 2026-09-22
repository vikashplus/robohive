DESC = '''
Parse mujoco (.mjl) logs\n
mjl format: http://www.mujoco.org/book/haptix.html#uiRecord
'''
import struct
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import click

# parse mjl binary logs into python dictionary
def parse_mjl_logs(read_filename, skipamount):
    with open(read_filename, mode='rb') as file:
        fileContent = file.read()
    headers = struct.unpack('iiiiiii', fileContent[:28])
    nq = headers[0]
    nv = headers[1]
    nu = headers[2]
    nmocap = headers[3]
    nsensordata = headers[4]
    nuserdata = headers[5]
    name_len = headers[6]
    name = struct.unpack(str(name_len) + 's', fileContent[28:28+name_len])[0]
    rem_size = len(fileContent[28 + name_len:])
    num_floats = int(rem_size/4)
    dat = np.asarray(struct.unpack(str(num_floats) + 'f', fileContent[28+name_len:]))
    recsz = 1 + nq + nv + nu + 7*nmocap + nsensordata + nuserdata
    if rem_size % recsz != 0:
        print("ERROR")
    else:
        dat = np.reshape(dat, (int(len(dat)/recsz), recsz))
        dat = dat.T

    time = dat[0,:][::skipamount] - 0*dat[0, 0]
    qpos = dat[1:nq + 1, :].T[::skipamount, :]
    qvel = dat[nq+1:nq+nv+1,:].T[::skipamount, :]
    ctrl = dat[nq+nv+1:nq+nv+nu+1,:].T[::skipamount,:]
    mocap_pos = dat[nq+nv+nu+1:nq+nv+nu+3*nmocap+1,:].T[::skipamount, :]
    mocap_quat = dat[nq+nv+nu+3*nmocap+1:nq+nv+nu+7*nmocap+1,:].T[::skipamount, :]
    sensordata = dat[nq+nv+nu+7*nmocap+1:nq+nv+nu+7*nmocap+nsensordata+1,:].T[::skipamount,:]
    userdata = dat[nq+nv+nu+7*nmocap+nsensordata+1:,:].T[::skipamount,:]

    data = dict(nq=nq,
               nv=nv,
               nu=nu,
               nmocap=nmocap,
               nsensordata=nsensordata,
               name=name,
               time=time,
               qpos=qpos,
               qvel=qvel,
               ctrl=ctrl,
               mocap_pos=mocap_pos,
               mocap_quat=mocap_quat,
               sensordata=sensordata,
               userdata=userdata,
               logName = read_filename
               )
    return data

# visualize parsed logs
def viz_parsed_mjl_logs(data):
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(data['time'], data['qpos'])
    axarr[0].set_ylabel('qpos')
    axarr[0].set_title(data['logName'])
    axarr[1].plot(data['time'], data['ctrl'])
    axarr[1].set_ylabel('ctrl')
    axarr[1].set_xlabel('time')
    plt.savefig(data['logName'][:-4]+".png")
    print(data['logName'][:-4]+".png saved")



# MAIN =========================================================
@click.command(help=DESC)
@click.option('--log', '-l', type=str, help='.mjl log to parse', required= True)
@click.option('--skip', '-s', type=int, help='number of frames to skip (1:no skip)', default=1)
@click.option('--plot', '-p', type=bool, help='plot parsed logs', default=False)
def main(log, skip, plot):
    print("Loading log file: %s" % log)
    data = parse_mjl_logs(log, skip)
    print("file successfully parsed")


    if(plot):
        print("plotting data")
        viz_parsed_mjl_logs(data)


if __name__ == '__main__':
    main()

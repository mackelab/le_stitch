import numpy as np
import bz2

data_path = '/nobackup/turaga/jane/for_m/dOMR0_20150414_112406/'
#data_path = '/home/mackelab/data/dOMR0_20150414_112406/'
results_path = '/groups/turaga/home/nonnenmacherm/data/dOMR0_20150414_112406/raw/'

T = 1200  # number of frames
Ts = np.arange(0, T, 1) 

nz, nx, ny = 41, 1024, 2048 # a word of caution: x,y,z in phicsal
pz = nx * ny                #

for z in range(nz):
    y = np.memmap(results_path+'y_z'+("%02d" % z), dtype=np.float16, mode='w+', shape=(T,pz))


for idx_t in range(len(Ts)):

    print(str(idx_t) + '/' + str(len(Ts)))
    
    t = Ts[idx_t]

    filename = data_path + 'TM' + ("%05d" % t) + '_CM0_CHN00.stack'

    dfile = bz2.BZ2File(filename,compresslevel=1)
    stack = np.frombuffer(dfile.read(), dtype=np.float16).reshape(nz,nx,ny)

    for z in range(nz):
        
        y = np.memmap(results_path+'y_z'+("%02d" % z), dtype=np.float16, mode='r+', shape=(T,pz))        
        y[idx_t,:] = stack[z,:,:].reshape(-1)
        del y # releases RAM, forces flush to disk
        
    del stack
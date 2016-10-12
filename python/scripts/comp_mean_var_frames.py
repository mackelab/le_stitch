import numpy as np
from scipy import stats 

data_path = '/groups/turaga/home/nonnenmacherm/data/dOMR0_20150414_112406/raw/'
tmp_path = '/groups/turaga/home/nonnenmacherm/data/dOMR0_20150414_112406/fits/'
nx, ny, nz = 41, 1024, 2048
px = ny * nz
T = 1200

y_means = np.zeros((nx,ny,nz))
y_vars  = np.zeros((nx,ny,nz))

for z in range(nx):
    
    print(str(z+1) + '/' + str(nx))
    
    # load data for current imaging plane
    data_file = data_path+'y_z'+("%02d" % z)
    y = np.memmap(data_file, dtype=np.float32, mode='r', shape=(T,px))    
    y_zscored = np.memmap(data_file+'_zsc', dtype=np.float32, mode='w+', shape=(T,px))    
    
    mean = np.zeros(px)
    var  = np.zeros(px)
    
    chunksize = 100000
    for i in range(px//chunksize):
        print(' - ' + str(i+1) + '/' + str(px//chunksize))
        
        idx = range(i*chunksize, (i+1)*chunksize)
        mean[idx] = np.mean(y[:,idx], axis=0)
        var[idx] = np.var(y[:,idx], axis=0)
        
        y_zscored[:,idx] = (y[:,idx] - mean[idx]) / var[idx]
        
        del y_zscored
        y_zscored = np.memmap(data_file+'_zsc', dtype=np.float32, mode='r+', shape=(T,px))    
        del y
        y = np.memmap(data_file, dtype=np.float32, mode='r', shape=(T,px))    
        
    idx = range((px//chunksize)*chunksize, px)
    mean[idx] = np.mean(y[:,idx], axis=0)
    var[idx] = np.var(y[:,idx], axis=0)   
    
    y_zscored[:,idx] = y_zscored[:,idx] = (y[:,idx] - mean[idx]) / var[idx]
    del y_zscored
    del y
    
    y_means[z,:,:] = mean.reshape(ny, nz)
    y_vars[z,:,:]  =  var.reshape(ny, nz)
    
    np.save(tmp_path + 'y_means', {'y_means': y_means})    
    np.save(tmp_path + 'y_var', {'y_vars': y_vars})


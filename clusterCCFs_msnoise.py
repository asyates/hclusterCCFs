from scipy import signal
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform 
from scipy.spatial import distance
from scipy import interpolate
from sklearn import preprocessing
from matplotlib.colors import ListedColormap
from matplotlib import cm
import pandas as pd
import numpy as np

def get_DMatrix(ccfs, params, distmethod, minlagwin, maxlagwin, norm=False, sides='B', dvvparams=[0.01,0.001]):
        
    if distmethod == 'euclid':
        D = compute_dmatrix_euclid(ccfs, params, minlagwin, maxlagwin, norm=norm, sides=sides) 

    elif distmethod == 'cc':
        D = compute_dmatrix_cc(ccfs, params, minlagwin, maxlagwin, norm=norm, sides=sides) 

    elif distmethod == 'ccstretch':
        D = compute_dmatrix_ccstretch(ccfs, params, minlagwin, maxlagwin, norm=norm, sides=sides, max_stretch=dvvparams[0], dvvstep=dvvparams[1])

    else:
        print('''Unrecognized clustering method, choose one of 'euclid', 'cc','ccstretch'.''') 
        return

    return D

def compute_dmatrix_euclid(ccfs, params, minlagwin, maxlagwin, norm=False, sides='B'):
    
    #get data corresponding to negative and positive lag times
    stack_n, stack_p = sliceCCFs(ccfs, params, minlagwin, maxlagwin, norm=norm)

    if sides == 'P':
        data = stack_p
    elif sides == 'N':
        data = stack_n
    else:
        data = np.concatenate((stack_n, stack_p), axis=1)


    D = distance.cdist(data, data, 'euclidean')
    D = np.around(D, decimals=6) #otherwise asymmetry issues sometimes

    return D

def compute_dmatrix_cc(ccfs, params, minlagwin, maxlagwin, norm=False, sides='B'):
    
    #get data corresponding to negative and positive lag times
    stack_n, stack_p = sliceCCFs(ccfs, params, minlagwin, maxlagwin, norm=norm)

    if sides == 'P':
        data = stack_p
    elif sides == 'N':
        data = stack_n
    else:
        data = np.concatenate((stack_n, stack_p), axis=1)

    #compute correlation coefficient matrix    
    corr = np.corrcoef(data)
    corr = (corr + corr.T)/2   # make symmetric
    np.fill_diagonal(corr, 1)  # put 1 on the diagonal

    #convert CC matrix to dissimilarity matrix
    D = 1 - np.abs(corr)
    D = np.around(D, decimals=6) #otherwise asymmetry issues sometimes

    has_nan = np.isnan(D)
    if has_nan.any() == True:
        print('WARNING: NaN value in distance matrix')

    return D

def compute_dmatrix_ccstretch(ccfs, params, minlagwin, maxlagwin, norm=False, sides='B', max_stretch=0.01, dvvstep=0.001): 

    # empty array that will store max values of CC
    maxCC = np.zeros((len(ccfs), len(ccfs))) 
    dvv_array = np.zeros((len(ccfs), len(ccfs)))

    #create list of all stretch values to apply
    stretch_values = np.arange(-max_stretch, max_stretch+dvvstep, dvvstep)
   
    #create lag time array using sampling rate and maxlag
    fs = params.cc_sampling_rate
    sampint = 1.0/fs
    maxlag = params.maxlag
 

    #get CCFs without stretching
    stack_n, stack_p = sliceCCFs(ccfs, params, minlagwin, maxlagwin, norm=norm)
    
    if sides == 'P':
        data_nostretch = stack_p
    elif sides == 'N':
        data_nostretch = stack_n
    else:
        data_nostretch = np.concatenate((stack_n, stack_p), axis=1)

    for i,ccf in enumerate(ccfs):
        CC_temp = np.zeros(len(ccfs))
        dvv_temp = np.zeros(len(ccfs))
        for value in stretch_values:
            
            #stretch CCF i.e introduce dv/v change
            ccf_stretched = stretchccf_stretchtime(ccf,value, maxlag, fs)
            
            #slice into negative and positive lag times
            stack_n, stack_p = sliceCCFs(np.array([ccf_stretched]), params, minlagwin, maxlagwin, norm=norm)

            if sides == 'P':
                data_stretched = stack_p
            elif sides == 'N':
                data_stretched = stack_n
            else:
                data_stretched = np.concatenate((stack_n, stack_p), axis=1)
            
            CC = 1 - distance.cdist(data_stretched,np.stack(data_nostretch), 'correlation')
            
            CC_temp = np.maximum(CC_temp, CC) #maybe for this, should only be looking at positive CC values
            
            dvv_temp = update_best_dvv(CC, CC_temp, value, dvv_temp)
            #print(dvv_temp)

                
        maxCC[i,:] = CC_temp
        maxCC[:,i] = CC_temp

        dvv_array[i,:] = dvv_temp
        dvv_array[:,i] = dvv_temp
    
    ##convert CC matrix to dissimilarity matrix
    np.fill_diagonal(maxCC, 1)  # put 1 on the diagonal
    D = 1 - np.abs(maxCC)

    return D

def update_best_dvv(corr, maxCC, stretch, dvv_array):

    for i in range(len(corr[0])):
        #print(i, corr[0][i], maxCC[0][i])
        #print(corr[0][i], maxCC[0][i])
        if corr[0][i] == maxCC[0][i]:
            dvv_array[i] = stretch
            #print(i)
    return dvv_array

def stretchdata(ccf_array, value, fs, maxlag):

    ccf_array_stretched = np.empty(len(ccf_array), dtype=object)

    for i, ccf in enumerate(ccf_array):
        ccf_stretched = stretchccf_stretchtime(ccf,value, maxlag, fs)
        ccf_array_stretched[i] = ccf_stretched

    return ccf_array_stretched

def stretchccf_stretchtime(ccf, dvv , maxlag, fs):

    scalefact = 1+(1*dvv)

    #maxlag = 120
    samprate = 1.0/fs

    lagtimes_orig = np.arange(-1*maxlag, maxlag+samprate, samprate)
    lagtimes_new = np.arange(-1*maxlag, maxlag+samprate, samprate)*scalefact

    
    #print(lagtimes_new)
    #print(len(lagtimes_new), len(ccf))

    f = interpolate.interp1d(lagtimes_new, ccf, fill_value='extrapolate')

    #fig0, ax0 = plt.subplots()

    #ax0.plot(lagtimes_orig, ccf)
    #ax0.plot(lagtimes_orig, f(lagtimes_orig))

    #plt.show()

    return f(lagtimes_orig)


def sliceCCFs(ccfs, params, minlagwin, maxlagwin, norm=False):

    #create lag time array using sampling rate and maxlag
    fs = params.cc_sampling_rate
    sampint = 1.0/fs
    maxlag = params.maxlag
    lagtimes = np.arange(-1*maxlag, maxlag+sampint, sampint)

    #get minimum and maximum index for snr windows, from minlagwin and maxlagwin
    minidx_psnr = np.abs(lagtimes-minlagwin).argmin()
    minidx_nsnr = np.abs(lagtimes-minlagwin*-1).argmin()
    maxidx_psnr = np.abs(lagtimes-maxlagwin).argmin()
    maxidx_nsnr = np.abs(lagtimes-maxlagwin*-1).argmin()

    #slice arrays to be just times of interest, both positive and negative side of CCF
    ccfs = np.vstack(np.array(ccfs))
      
    if norm == True: #normalise using maximum value
        max_values = np.max(np.abs(ccf_array),axis=1) #gets max value in each row/CCF
        ccfs = ccfs / max_values[:,None]

    stack_p = ccfs[:,minidx_psnr:maxidx_psnr+1]
    stack_n = ccfs[:,maxidx_nsnr:minidx_nsnr+1]
 
    return stack_n, stack_p 

def getCluster(day, all_days, labels, ccfs):

    #find index of chosen day in all days    
    try:
        idx = np.where(all_days == day)[0][0]
    except:
        print('error: check chosen day included in cluster results')

    labelidx = labels[idx]

    selectlabels = np.where(labels == labelidx)
    days_cluster = all_days[selectlabels]
    selectccfs = ccfs[selectlabels]

    return days_cluster, selectccfs
    


def plot_interferogram(ccfs, params, days, fig=None, ax=None, ax_cb=None, maxlag=120):

    if ax_cb == None:
        ax_cb = ax

    fs = params.cc_sampling_rate
    sampint = 1.0/fs
    maxlag_msnoise = params.maxlag
    lagtimes = np.arange(-1*maxlag_msnoise, maxlag_msnoise+sampint, sampint)


    df = pd.DataFrame(np.array(ccfs).real.tolist(), index=days, columns=lagtimes)
    df = df.dropna()

    #define the 99% percentile of data for visualisation purposes
    clim = df.mean(axis='index').quantile(0.99)

    print(fig)
    print(ax)
    if fig == None or ax == None:
        fig, ax = plt.subplots(figsize=(12,8))
        plot = True
    else:
        plot = False

    img = ax.pcolormesh(df.index, df.columns, df.values.T, vmin=-clim, vmax=clim, rasterized=True,cmap='seismic')
    #fig.colorbar(img, cax=ax_cb).set_label('')
  
    
    #plt.colorbar()
    #ax.set_title('Interferogram')
    ax.set_ylabel('Lag Time (s)')
    ax.set_ylim(maxlag*-1, maxlag)

    if plot == True:
        plt.show()



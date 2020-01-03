import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from correct_deepESN import ESN

# training/testing parameters
n_ens = 100
n_opt = 1 
threshold = 1000  #a choice (start with an initial threshold and slowly decrease it if necessary -> pocket)
#special case when threshold is big enough such that a given random seed will be used
valid = 10 # a choice

future = 610+valid # signal data used for validation
trainlen = 3690-valid # total number of available signal data =  trainlen (#data used for training) + valid (#data used for val)
trainbeg = 0

dt=0.01 #time step (known)

def F(soln):
    return soln-soln**3

def diff(x,dt):
    y=np.zeros((len(x),1))
    for i in range(len(x)-1):
        y[i]=(x[i+1]-x[i])/dt-F(x[i])
    return y

data_orig = pd.read_csv("xdata_eg1.csv",header=None)
data_orig = np.array(data_orig)
data_orig = data_orig[:,1]

forcing=diff(data_orig,dt)
data = forcing

#print info
print('Total number of signal data points given: ',trainlen+valid) #total given = this + 1
print('Training data starts from data point #: ', trainbeg)
print('Training data ends at data point #: ',trainlen)
print('Total number of data points used for validation: ', valid)
print('Total number of data points used for training: ',trainlen-trainbeg)
#trainlen -= 1

opred_training = []
oprediction = []

otest_error = np.zeros((n_ens,1))
oNum=int(trainlen+future); dtau=dt
osol=np.zeros((n_ens,oNum+1-trainbeg))

valid_error_rmse=np.zeros((n_ens,8,1,1,1))
seeds = np.zeros(valid_error_rmse.shape)
k=0
count = 0
max_iter = 1000

while k < n_ens and count <= max_iter:
    for h1 in range(8): 
        for h2 in range(1): 
            for h3 in range(1): 
                for h4 in range(1): 
                  

                    print("======================= Optimizing over a hyperparameter space ====================== ")
                    print('n_reservoir = ', 600+h1*20)
                    print('spectral radius = ', 0.7+h2*0.1)
                    print('sparsity = ', 0.1+h3*0.1)
                    print('noise = ', 0.001+0.001*int(h4))
    
                    esn = ESN(n_inputs = 1,
                              n_outputs = 1,
                              n_reservoir = [600+h1*20], 
                              n_layer= 1,
                              spectral_radius = [0.7+h2*0.1],
                              sparsity= [0.1+h3*0.1], 
                              noise=0.001+0.001*h4,
                              random_state= count+100)

                    fitt = esn.fit(np.ones(trainlen-trainbeg),data[trainbeg:trainlen], inspect = False)
                    predd =  esn.predict(np.ones(valid))
                    valid_error = np.sqrt(np.mean((predd.flatten() - data[trainlen:trainlen+valid])**2))

                    seeds[k,h1,h2,h3,h4] = count+100

                    print("validation error for reconstructed signal: \n"+str(valid_error))
                    valid_error_rmse[k,h1,h2,h3,h4] = valid_error

                   
    count += 1
    k_valid_error_rmse = valid_error_rmse[k,:,:,:,:]
    print('=============> index of min validation errors', np.unravel_index(np.argmin(k_valid_error_rmse, axis=None), k_valid_error_rmse.shape))
    ind_vec = np.unravel_index(np.argmin(k_valid_error_rmse, axis=None), k_valid_error_rmse.shape)
    min_error = k_valid_error_rmse[ind_vec]    
    
    if min_error <= threshold:
        print('min_error = ', min_error)
        print('opt n_reservoir = ', 600+int(ind_vec[0]*20))
        print('opt spectral radius = ', 0.7+ind_vec[1]*0.1)
        print('opt sparsity = ', 0.1+ind_vec[2]*0.1)
        print('opt noise = ', 0.001+0.001*ind_vec[3])

        opt_ind = np.array(ind_vec).T
        opt_seeds = seeds[k,ind_vec[0],ind_vec[1],ind_vec[2],ind_vec[3]]
        print('Seed number: ', opt_seeds)

        oesn = ESN(n_inputs = 1,
                   n_outputs = 1,
                   n_reservoir = [600+int(opt_ind[0]*20)],
                   n_layer= 1,
                   spectral_radius = [0.7+opt_ind[1]*0.1], 
                   sparsity= [0.1+opt_ind[2]*0.1], 
                   noise=0.001+0.001*opt_ind[3], 
                   random_state= int(opt_seeds)) 
        opred_training.append(oesn.fit(np.ones(trainlen-trainbeg),data[trainbeg:trainlen], inspect = False))
        oprediction.append(oesn.predict(np.ones(future)))

        otest_error[k] = np.sqrt(np.mean((oprediction[k].flatten() - data[trainlen:trainlen+future])**2))
        print("----------------------------------------------------------------------")
        print("=====================================================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Run #: ", k)
        print("test error for predicted signal: \n"+str(otest_error[k]))
        print("----------------------------------------------------------------------")

        oq = oprediction[k] 
        oq=oq.reshape((oq.shape[0],1))
        oq=np.concatenate((forcing[trainbeg:trainlen],oq),axis=0)

        osol[k,0]=data_orig[trainbeg]
        for n in range(trainbeg,oNum):
            k1=dtau*(F(osol[k,n-trainbeg]))
            k2=dtau*F(osol[k,n-trainbeg]+k1/2)
            k3=dtau*F(osol[k,n-trainbeg]+k2/2)
            k4=dtau*F(osol[k,n-trainbeg]+k3)
            osol[k,n+1-trainbeg]=osol[k,n-trainbeg]+(k1+2*k2+2*k3+k4)/6+dtau*oq[n-trainbeg] 
        k+=1
        
np.savetxt('RSPA19_Ex1_3690-10_100ens_fin.csv',np.c_[osol], delimiter=',')   

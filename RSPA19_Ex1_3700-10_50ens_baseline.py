import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from correct_deepESN import ESN

# training/testing parameters
n_ens = 50
n_opt = 1 
threshold = 1000  #a choice (start with an initial threshold and slowly decrease it if necessary -> pocket)
#special case when threshold is big enough such that a given random seed will be used
valid = 10 # a choice

future = 600+valid # # signal data used for validation
trainlen = 3700-valid # total number of available signal data =  trainlen (#data used for training) + valid (#data used for val)
trainbeg = 0

data_orig = pd.read_csv("xdata_eg1.csv",header=None)
data_orig = np.array(data_orig)
data_orig = data_orig[:,1]

data = data_orig

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
osol=np.zeros((n_ens,future))

test_error_rmse=np.zeros((n_ens,5,1,1,1))
seeds = np.zeros(test_error_rmse.shape)
k=0
count = 0
max_iter = 1000

while k < n_ens and count <= max_iter:
    for h1 in range(5): 
        for h2 in range(1): 
            for h3 in range(1): 
                for h4 in range(1): 
                    Num=int(trainlen+valid)
                    sol=np.zeros((Num+1-trainbeg,1))

                    print("======================= Optimizing over a hyperparameter space ====================== ")
                    print('n_reservoir = ', 660+h1*20)
                    print('spectral radius = ', 0.7+h2*0.05)
                    print('sparsity = ', 0.1+h3*0.1)
                    print('noise = ', 0.001+0.0005*int(h4))
    
                    esn = ESN(n_inputs = 1,
                              n_outputs = 1,
                              n_reservoir = [660+h1*20], 
                              n_layer= 1,
                              spectral_radius = [0.7+h2*0.05],
                              sparsity= [0.1+h3*0.1], 
                              noise=0.001+0.0005*h4,
                              random_state= count+100)

                    fitt = esn.fit(np.ones(trainlen-trainbeg),data[trainbeg:trainlen], inspect = False)
                    predd =  esn.predict(np.ones(valid))
                    test_error = np.sqrt(np.mean((predd.flatten() - data[trainlen:trainlen+valid])**2))

                    seeds[k,h1,h2,h3,h4] = count+100

                    print("validation error for reconstructed signal: \n"+str(test_error))
                    test_error_rmse[k,h1,h2,h3,h4] = test_error

                
    count += 1
    k_test_error_rmse = test_error_rmse[k,:,:,:,:]
    print('=============> index of min validation errors', np.unravel_index(np.argmin(k_test_error_rmse, axis=None), k_test_error_rmse.shape))
    ind_vec = np.unravel_index(np.argmin(k_test_error_rmse, axis=None), k_test_error_rmse.shape)
    min_error = k_test_error_rmse[ind_vec]    
    
    if min_error <= threshold:
        print('min_error = ', min_error)
        print('opt n_reservoir = ', 660+int(ind_vec[0]*20))
        print('opt spectral radius = ', 0.7+ind_vec[1]*0.05)
        print('opt sparsity = ', 0.1+ind_vec[2]*0.1)
        print('opt noise = ', 0.001+0.0005*ind_vec[3])

        opt_ind = np.array(ind_vec).T
        opt_seeds = seeds[k,ind_vec[0],ind_vec[1],ind_vec[2],ind_vec[3]]
        print('Seed number: ', opt_seeds)

        oesn = ESN(n_inputs = 1,
                   n_outputs = 1,
                   n_reservoir = [660+int(opt_ind[0]*20)],
                   n_layer= 1,
                   spectral_radius = [0.7+opt_ind[1]*0.05], 
                   sparsity= [0.1+opt_ind[2]*0.1], 
                   noise=0.001+0.0005*opt_ind[3], 
                   random_state= int(opt_seeds)) 
        opred_training.append(oesn.fit(np.ones(trainlen-trainbeg),data[trainbeg:trainlen], inspect = False))
        oprediction.append(oesn.predict(np.ones(future)))

        otest_error[k] = np.sqrt(np.mean((oprediction[k].flatten() - data[trainlen:trainlen+future])**2))
        print("----------------------------------------------------------------------")
        print("=====================================================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Run #: ", k)
        print("test error for predicted signal: \n"+str(otest_error[k]))
        print("----------------------------------------------------------------------")

        osol[k,:] = oprediction[k].reshape((future,))
    
        k+=1
        
np.savetxt('RSPA19_Ex1_3700-10_50ens_baseline.csv',np.c_[osol], delimiter=',')   

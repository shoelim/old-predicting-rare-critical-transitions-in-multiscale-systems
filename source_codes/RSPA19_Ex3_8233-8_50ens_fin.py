import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from correct_deepESN import ESN

# training/testing parameters
n_ens = 100
n_opt = 1 
threshold = 1000 #0.05 #a choice (start with an initial threshold and slowly decrease it if necessary -> pocket)
#special case when threshold is big enough such that a given random seed will be used

valid = 8 # a choice
future = 37+valid # # signal data used for validation
trainlen = 8233-valid # total number of available signal data =  trainlen (#data used for training) + valid (#data used for val)
trainbeg = 200

A=0.1
dt=0.01

def F(x):
    return x*(1-x)*(1+x)*(x-2)*(x+2)

def diff(x,dt):
    y=np.zeros((len(x),1))
    for i in range(len(x)-1):
        y[i]=(x[i+1]-x[i])/dt-F(x[i])-A*np.cos(12*np.pi*dt*i)
    return y

def Fn(x,t):
    return x*(1-x)*(1+x)*(x-2)*(x+2)+A*np.cos(12*np.pi*t)

data_orig = pd.read_csv("xdata_eg3.csv",header=None)
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

test_error_rmse=np.zeros((n_ens,7,1,1,1))
seeds = np.zeros(test_error_rmse.shape)
k=0
count = 0
max_iter = 1000

while k < n_ens and count <= max_iter:
    for h1 in range(7): 
        for h2 in range(1): 
            for h3 in range(1): 
                for h4 in range(1): 
                    Num=int(trainlen+valid)
                    sol=np.zeros((Num+1-trainbeg,1))

                    print("======================= Optimizing over a hyperparameter space ====================== ")
                    print('n_reservoir in each layer = ', 500+h1*50)
                    print('spectral radius = ', 0.99+h2*0.05)
                    print('sparsity = ', 0.1+h3*0.05)
                    print('noise = ', 0.0005+0.001*int(h4))
    
                    esn = ESN(n_inputs = 1,
                              n_outputs = 1,
                              n_reservoir = [500+h1*50], #[200,200,200], [150,150,150] 
                              n_layer= 1,
                              spectral_radius = [0.95+h2*0.05], #[0.6,0.7,0.8], [0.55,0.65,0.75]
                              sparsity= [0.1+h3*0.05], #[0.05,0.05,0.05], [0.08,0.08,0.08]
                              noise=0.003+0.001*h4, #0.003, 0.002 
                              #noise_in_prediction=False,
                              n_transient=100,
                              #noise_option = 1,
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
        print('opt n_reservoir in each layer = ', 500+int(ind_vec[0]*50))
        print('opt spectral radius = ', 0.99+ind_vec[1]*0.05)
        print('opt sparsity = ', 0.1+ind_vec[2]*0.05)
        print('opt noise = ', 0.0005+0.001*ind_vec[3])

        opt_ind = np.array(ind_vec).T
        opt_seeds = seeds[k,ind_vec[0],ind_vec[1],ind_vec[2],ind_vec[3]]
        print('Seed number: ', opt_seeds)

        oesn = ESN(n_inputs = 1,
                   n_outputs = 1,
                   n_reservoir = [500+int(opt_ind[0]*50)],
                   n_layer= 1,
                   spectral_radius = [0.95+opt_ind[1]*0.05], 
                   sparsity= [0.1+opt_ind[2]*0.05], 
                   noise=0.003+0.001*opt_ind[3], 
                   #noise_in_prediction=False,
                   n_transient=100,
                   #noise_option = 1,
                   random_state= int(opt_seeds)) 
        opred_training.append(oesn.fit(np.ones(trainlen-trainbeg),data[trainbeg:trainlen], inspect = False))
        oprediction.append(oesn.predict(np.ones(future)))

        otest_error[k] = np.sqrt(np.mean((oprediction[k].flatten() - data[trainlen:trainlen+future])**2))
        print("----------------------------------------------------------------------")
        print("=====================================================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Run #: ", k)
        print("test error for predicted signal: \n"+str(otest_error[k]))
        print("----------------------------------------------------------------------")

        oq = oprediction[k] 
        
        t_res=np.linspace(trainlen,(trainlen+future),future)
        #plt.figure(figsize=(22,5))
        #plt.plot(t_res,oq,'r-')
        #plt.plot(t_res,forcing_true[trainlen:trainlen+future])
        #plt.legend(['predicted','actual'])
        #plt.title('reconstructed driving signal')
        #plt.show()
        
        oq=oq.reshape((oq.shape[0],1))
        oq=np.concatenate((forcing[trainbeg:trainlen],oq),axis=0)

        osol[k,0]=data_orig[trainbeg]
        for n in range(trainbeg,oNum):
            k1=dtau*(Fn(osol[k,n-trainbeg],dtau*n))
            k2=dtau*Fn(osol[k,n-trainbeg]+k1/2,dtau*n + dtau/2)
            k3=dtau*Fn(osol[k,n-trainbeg]+k2/2,dtau*n + dtau/2)
            k4=dtau*Fn(osol[k,n-trainbeg]+k3,dtau*n + dtau)
            osol[k,n+1-trainbeg]=osol[k,n-trainbeg]+(k1+2*k2+2*k3+k4)/6+dtau*oq[n-trainbeg] 
        k+=1
        
np.savetxt('RSPA19_Ex3_8233-8_100ens.csv',np.c_[osol], delimiter=',')  
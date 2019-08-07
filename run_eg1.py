# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:40:39 2019

@author: Soon Hoe Lim
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from deepESN import ESN

#number of runs for averaging
n_ens = 10  #100

def F(soln):
    return soln-soln**3

def diff(x,dt):
    y=np.zeros((len(x),1))
    for i in range(len(x)-1):
        y[i]=(x[i+1]-x[i])/dt-F(x[i])
    return y

dt=0.01

data_orig = pd.read_csv("xdata_eg1.csv",header=None)
data_orig = np.array(data_orig)
data_orig = data_orig[:,1]
#print(data_orig.shape)

forcing=diff(data_orig,dt)
data = forcing

n_reduce = 1 #6
reduce_size = 500 #1000
mean_rmse = np.zeros(n_reduce)
std_rmse = np.zeros(n_reduce)

for k in range(n_reduce):
    #training/testing parameters
    trainbeg = k*reduce_size #4000 for 2nd jump 
    trainlen = 3700 #3690,3680,3670; 7080
    future = 600 #610,620,630; 935

    #print info
    print('Training data starts from data point #: ', trainbeg)
    print('Training data ends at data point #: ',trainlen)
    print('Total number of data points used for training: ',trainlen-trainbeg)
    print('Number of data points into the future to be predicted: ', future)
    
    test_error = np.zeros((n_ens,1)) #test error for reconstructed signal
    Num=int(trainlen+future); dtau=dt
    sol=np.zeros((n_ens,Num+1-trainbeg))

    pred_training = []
    prediction = []

    for i in range(n_ens):
        esn = ESN(n_inputs = 1,
                  n_outputs = 1,
                  n_reservoir = [720], #[720]-immediate, [700]-10ahead&20ahead&30ahead; [1000]-2ndjump using whole, [720]-2ndjump with fewer data
                  n_layer= 1,
                  nonlin = 1,
                  spectral_radius = [0.7], #[0.7], [0.7]; [0.8], [0.75]
                  teacher_forcing = True,
                  sparsity= [0.1], #[0.1], [0.1]; [0.1], [0.1]
                  noise=0.001,
                  #ilent = False,
                  random_state= i+100) 
        pred_training.append(esn.fit(np.ones(trainlen-trainbeg),data[trainbeg:trainlen], inspect = False))
        prediction.append(esn.predict(np.ones(future)))
        #print(prediction.shape)
        test_error[i] = np.sqrt(np.mean((prediction[i].flatten() - data[trainlen:trainlen+future])**2))
        print(i)
        print("test error for reconstructed signal: \n"+str(test_error[i]))
    
        q = prediction[i] 
        q=q.reshape((q.shape[0],1))
        q=np.concatenate((forcing[trainbeg:trainlen],q),axis=0)
    
        sol[i,0]=data_orig[trainbeg]

        for n in range(trainbeg,Num):
            k1=dtau*(F(sol[i,n-trainbeg]))
            k2=dtau*F(sol[i,n-trainbeg]+k1/2)
            k3=dtau*F(sol[i,n-trainbeg]+k2/2)
            k4=dtau*F(sol[i,n-trainbeg]+k3)
            sol[i,n+1-trainbeg]=sol[i,n-trainbeg]+(k1+2*k2+2*k3+k4)/6+dtau*q[n-trainbeg] 
            
        #get statistics of rmse for testing part of reconstructed signal
        #print(np.mean(test_error))
        #print(np.std(test_error))
        #plt.plot(test_error)
        #plt.show()
        
    #####################################################################################################      
    #visualize results
    t_tr=np.linspace(trainbeg,trainlen,trainlen-trainbeg)
    t_res=np.linspace(trainlen,(trainlen+future),future)

    plt.rcParams['axes.facecolor']='white'

    #####################uncomment below to see reconstructed forcing and error##########################
    #forcing_data_orig = forcing
    #plt.figure(figsize=(12,4),facecolor='white')
    #plt.plot(forcing[:]) #-forcing_true[:8000])
    #plt.grid(False)
    #plt.show()

    #plt.figure(figsize=(12,4))
    #plt.plot((forcing[:]-forcing_true[:])/forcing_true[:])
    #plt.grid(b=None)
    #plt.show()

    #plt.hist(forcing[:])
    #plt.show()

    #reconstructed driving signal
    #plt.figure(figsize=(12,4))
    #plt.plot(t_tr, q[:trainlen])
    #plt.plot(t_res,q[trainlen:trainlen+future],'r-')
    #plt.plot(t_res,forcing_data_orig[trainlen:trainlen+future])
    #plt.legend(['training','predicted','actual'])
    #plt.title('reconstructed driving signal')
    #plt.show()

    #sol = pd.read_csv("out_eg1_05_3layer.csv",header=None)
    #sol = np.array(sol)

    #main result
    ax1=plt.figure(figsize=(6,3))
    plt.plot(t_tr,data_orig[trainbeg:trainlen],'r^')
    plt.plot(t_res,data_orig[trainlen:trainlen+future],'r^')
    solp=[]
    for i in range(n_ens):
        solp.append(sol[i,trainlen-trainbeg:trainlen+future-trainbeg])
        plt.plot(t_res,solp[i],alpha=0.2)
    plt.plot(t_res, np.mean(solp,axis=0),'b-o')
    ax1.text(0.1, 0.96,'(a)', fontsize=12, verticalalignment='top')
    #plt.grid(b=None)
    #plt.legend(['training','predicted','actual'])
    #plt.title('position (slow variable)')
    plt.show()

    #pathwise metric:
    #error for predicted position
    ax2=plt.figure(figsize=(6,3))
    error=[]
    rmse_vec=np.zeros(n_ens)
    for i in range(n_ens):
        diff=sol[i,trainlen-trainbeg:trainlen+future-trainbeg]-data_orig[trainlen:trainlen+future]
        rmse=np.sqrt(np.mean(diff**2))
        error.append(diff)
        rmse_vec[i]=rmse
        plt.plot(t_res, error[i],alpha=0.2)
    plt.plot(t_res, np.mean(error,axis=0),'b-o')
    ax2.text(0.1, 0.96,'(c)', fontsize=12, verticalalignment='top')
    #plt.grid(b=None)
    #plt.title('error for predicted position')
    plt.show()

    #std dev for predicted positions
    ax3=plt.figure(figsize=(6,3))
    stdev=np.std(error,axis=0)
    plt.plot(t_res, stdev,'b-o')
    ax3.text(0.1, 0.96,'(d)', fontsize=12, verticalalignment='top')
    #plt.grid(b=None)
    #plt.title('std dev for predicted position')
    plt.show()

    #coarse-grained metric:
    #statistics of the rmses (w.r.t. ensembles) for predicted position on the prediction interval
    mean_rmse[k] = np.mean(rmse_vec)
    std_rmse[k] =  np.std(rmse_vec)
    print('Mean of rmse:', mean_rmse[k])
    print('Std deviation of rmse:', std_rmse[k])

    #true position, multiple predicted positions, the averaged prediction and the 90 percent confidence interval
    sonn=[]
    ax4=plt.figure(figsize=(6,3))
    plt.plot(t_res,data_orig[trainlen:trainlen+future],'r^')
    for i in range(n_ens):
        sonn.append(sol[i,trainlen-trainbeg:trainlen+future-trainbeg])
        plt.plot(t_res, sonn[i],alpha=0.2)
    stderr = sem(sonn,axis=0)  #std error of the mean (sem) provides a simple measure of uncertainty in a value
    #Remark: Confidence interval is calculated assuming the samples are drawn from a Gaussian distribution
    #Justification: As the sample size tends to infinity the central limit theorem guarantees that the sampling 
    #               distribution of the mean is asymptotically normal
    plt.plot(t_res,np.mean(sonn,axis=0),'b-o')
    y1=np.mean(sonn,axis=0)-1.645*stderr
    y2=np.mean(sonn,axis=0)+1.645*stderr
    plt.plot(t_res,y1,'--')
    plt.plot(t_res,y2,'--')
    plt.fill_between(t_res, y1, y2, facecolor='blue', alpha=0.2)
    ax4.text(0.1, 0.96,'(b)', fontsize=12, verticalalignment='top')
    #plt.grid(False)
    #plt.title('true position, multiple predicted positions, the averaged prediction and the 90 percent confidence interval')
    plt.show()

#plot the relationship between number of training data used and statistics of rmse at a fixed training setting
plt.rcParams['axes.facecolor']='white'
numm=np.arange(trainlen,trainlen-n_reduce*reduce_size,-reduce_size)
ax = plt.figure(figsize=(6,3))
plt.plot(numm,mean_rmse,'b-o')
plt.plot(numm,std_rmse,'r--o')
plt.legend(['mean of the rmse','std of the rmse'],loc='upper right')
plt.xlabel('Number of data points used for training',fontsize=12)
ax.text(0.1, 0.96,'(?)', fontsize=12, verticalalignment='top')  #change label (?) according to experiment trials
plt.show()


#############################################################################################
#using naive direct method

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from deepESN import ESN

n_ens = 10 #100

data_orig = pd.read_csv("xdata_eg1.csv",header=None)
data_orig = np.array(data_orig)
data_orig = data_orig[:,1]
print(data_orig.shape)

data = data_orig

#training/testing parameters
#training data always starts from time zero here
trainlen = 3700
future = 315

test_error = np.zeros((future,1))
x0=-1.5
Num=int(trainlen+future); dtau=dt
sol=np.zeros((n_ens,Num+1))

pred_training = []
prediction = []

for i in range(n_ens):
    esn = ESN(n_inputs = 1,
              n_outputs = 1,
              n_reservoir = [600], 
              n_layer= 1,
              nonlin = 1,
              spectral_radius = [0.7], 
              teacher_forcing = True,
              sparsity= [0.1], 
              noise=0.001,
              silent = True,
              random_state=i+100) 
    pred_training.append(esn.fit(np.ones(trainlen),data[:trainlen], inspect = False))
    prediction.append(esn.predict(np.ones(future)))
    #print(prediction.shape)
    test_error[i] = np.sqrt(np.mean((prediction[i].flatten() - data[trainlen:trainlen+future])**2))
    print(i)
    print("test error: \n"+str(test_error[i]))

#############################################################################################
#visualize results 
t_tr=np.linspace(0,trainlen,trainlen)
t_res=np.linspace(trainlen,(trainlen+future),future)

plt.rcParams['axes.facecolor']='white'

#main result
ax5 = plt.figure(figsize=(6,3))
plt.plot(t_tr,data_orig[:trainlen],'r^')
plt.plot(t_res,data_orig[trainlen:trainlen+future],'r^')
solp=[]
for i in range(n_ens):
    solp.append(prediction[i])
    plt.plot(t_res,solp[i],alpha=0.3)
plt.plot(t_res, np.mean(solp,axis=0),'b-o')
#plt.grid(b=None)
#plt.legend(['training','predicted','actual'])
#plt.title('position (slow variable)')
ax5.text(0.1, 0.96,'(a)', fontsize=12, verticalalignment='top')
plt.show()

#error for predicted position
ax6 = plt.figure(figsize=(6,3))
error=[]
print(prediction[0].shape)
print(data_orig[trainlen:trainlen+future].shape)
for i in range(n_ens):
    error.append(prediction[i]-data_orig[trainlen:trainlen+future].reshape((future,1)))
    plt.plot(t_res,error[i],alpha=0.3)
plt.plot(t_res,np.mean(error,axis=0),'b-o')
ax6.text(0.1, 0.96,'(c)', fontsize=12, verticalalignment='top')
#plt.grid(b=None)
#plt.title('error for predicted position')
plt.show()

#std dev for predicted positions
ax7 = plt.figure(figsize=(6,3))
stdev=np.std(error,axis=0)
plt.plot(t_res,stdev,'b-o')
#plt.grid(b=None)
#plt.title('std dev for predicted position')
ax7.text(0.1, 0.96,'(d)', fontsize=12, verticalalignment='top')
plt.show()

#true position, multiple predicted positions, the averaged prediction and the 90 percent confidence interval
sonn=[]
ax8=plt.figure(figsize=(6,3))
plt.plot(t_res,data_orig[trainlen:trainlen+future],'r^')
for i in range(n_ens):
    sonn.append(prediction[i])
    plt.plot(t_res, sonn[i],alpha=0.3)
stderr = sem(sonn,axis=0) 
plt.plot(t_res,np.mean(sonn,axis=0),'b-o')
y1=np.mean(sonn,axis=0)-1.645*stderr
y2=np.mean(sonn,axis=0)+1.645*stderr
y1=y1.reshape((y1.shape[0],))
y2=y2.reshape((y2.shape[0],))
plt.plot(t_res,y1,'--')
plt.plot(t_res,y2,'--')
plt.fill_between(t_res, y1, y2, facecolor='blue', alpha=0.3)
ax8.text(0.1, 0.96,'(b)', fontsize=12, verticalalignment='top')
#plt.grid(False)
#plt.title('true position, multiple predicted positions, the averaged prediction and the 90 percent confidence interval')
plt.show()
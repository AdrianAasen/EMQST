import numpy as np
import os
import uuid
from datetime import datetime
from scipy.stats import unitary_group
import qutip as qt
from joblib import Parallel, delayed
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import sqrtm
import time

import scipy

#from support_functions import *
#from nct import *
#from qst import *


from EMQST_lib import support_functions as sf
from EMQST_lib import measurement_functions as mf
from EMQST_lib import dt, emqst
from EMQST_lib.qst import QST
from EMQST_lib.povm import POVM




def main():
    measurement_test()
   # BME_test()
   # noise_test()

   # nct_test()
    two_qubit_test()
    
    dt_noise_singe_qubit_test()
    print("All tests was completed successfully!")
    return 1

def measurement_test():
    np.random.seed(0)
    POVMset1=np.array([[[1,0],[0,0]],[[0,0],[0,1]]],dtype=complex)
    POVMset2=1/2*np.array([[[1,-1j],[1j,1]],[[1,1j],[-1j,1]]],dtype=complex)
    iniPOVM1=POVM(POVMset1,np.array([[[0,0],[np.pi,0]]]))
    iniPOVM2=POVM(POVMset2,np.array([[[np.pi/2,np.pi/2],[np.pi/2,3*np.pi/2]]]))   
    #POVM1=np.array([iniPOVM1])
    #POVM2=np.array([iniPOVM2])
    #POVM_list=np.array([iniPOVM1,iniPOVM2])
    # Prepare x state
    rho=np.array([[1/2,1/2],[1/2,1/2]],dtype=complex)
    n=100
    outcome1=mf.simulated_measurement(n,iniPOVM1,rho)

    assert np.array_equal(outcome1,np.array([1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1,
                                            0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0,
                                            0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0,
                                            1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0,
                                            1, 0, 1, 0],dtype=float))
    outcome2=mf.simulated_measurement(n,iniPOVM2,rho)

    assert np.array_equal(outcome2,np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                            0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1,
                                            1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0,
                                            0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0,
                                            0, 0, 0, 0],dtype=float))
    return 1

def nct_test():
    # Generate new dictionary
    now=datetime.now()
    now_string = now.strftime("%Y-%m-%d_%H-%M-%S_")
    dir_name= now_string+str(uuid.uuid4())
    data_path=f'results/{dir_name}'
    os.mkdir(data_path)



    n_qubits=1
    n_QST_shots=10**5
    n_calibration_shots=10**5
    exp_dictionary={}
    list_of_true_states=np.array([[[1/2,1/2],[1/2,1/2]]],dtype=complex)
    
    uncorrected_infidelity, corrected_infidelity,rho_estm=emqst.emqst(1,n_QST_shots,n_calibration_shots,list_of_true_states,data_path )
    shots=np.arange(len(uncorrected_infidelity[0]))
    plt.plot(shots,corrected_infidelity[0])
    plt.plot(shots,uncorrected_infidelity[0])
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(100,len(shots))
    plt.ylim(10**(-5),10**(-0))
    plt.ylabel('Mean Infidelity')
    plt.xlabel('Number of shots')


    
    with open(f'{data_path}/infidelity.npy','wb') as f:
        np.save(f,uncorrected_infidelity)
        np.save(f,corrected_infidelity )
    with open(f'{data_path}/experimental_settings.npy','wb') as f:
        np.save(f,exp_dictionary)

    plt.savefig(f'{data_path}/Test.png')




def BME_test():
    #np.random.seed(0)
    n_qubits=1
    noise_mode=4
    POVM_list=POVM.generate_Pauli_POVM(n_qubits)
    noisy_POVM=np.array([POVM.generate_noisy_POVM(povm,noise_mode) for povm in POVM_list])
    
    # Prepare x state
    #true_state_list=np.array([[[1/2,1/2],[1/2,1/2]]],dtype=complex)
    n_averages=10
    true_state_list=np.array([sf.generate_random_pure_state(1) for _ in range(n_averages)])
    n_shots_each=10**5
    test_QST=QST(noisy_POVM,true_state_list,n_shots_each,1,n_cores=5,noise_corrected_POVM_list=POVM_list)
    test_QST.generate_data()
    test_QST.perform_BME()
    
    #cutoff
    print(test_QST.infidelity)
    #print(test_QST.infidelity)
    shots=np.arange(len(test_QST.infidelity[0]))
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(8,6))
    plt.plot(shots,test_QST.infidelity[0])
    
    test_QST.perform_BME(use_corrected_POVMs=True)
    plt.plot(shots,test_QST.infidelity[0])
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(100,len(shots))
    plt.ylim(10**(-5),10**(-0))
    plt.ylabel('Mean Infidelity')
    plt.xlabel('# of shots')
    plt.savefig("Test.png")
    #a=np.array([1,2,3,4])
    #b=[1,2,3,0]
    #print(a[b])

    return 1

def func1():
    print("Hello")
    return 10


def dt_noise_singe_qubit_test():
    print("DT noise test:")
    np.random.seed(0)
    POVMset=1/2*np.array([[[1,-1j],[1j,1]],[[1,1j],[-1j,1]]],dtype=complex)#np.array([[[1,0],[0,0]],[[0,0],[0,1]]],dtype=complex)#
    iniPOVM=POVM(POVMset,np.array([[[0,0],[np.pi,0]]]))
    bool_exp_meaurement=False
    expDict={}
    calibration_angles=np.array([[[np.pi/2,0]],[[np.pi/2,np.pi]],
                        [[np.pi/2,np.pi/2]],[[np.pi/2,3*np.pi/2]],
                        [[0,0]],[[np.pi,0]]])
    calibration_states=np.array([sf.get_density_matrix_from_angles(angle) for angle in calibration_angles])



    nShots=10**4
    start = time.time()

    for i in range(4):
        noise_mode=i+1
        noisy_POVM=POVM.generate_noisy_POVM(iniPOVM,noise_mode)
        #print(noisy_POVM.get_POVM())
        corrPOVM=dt.device_tomography(1, nShots, noisy_POVM,calibration_states,bool_exp_meaurement,expDict,iniPOVM)
        #print(corrPOVM.get_POVM())
        print(f'Distance between reconstructed and noisy POVM: {sf.POVM_distance(corrPOVM.get_POVM(),noisy_POVM.get_POVM())}')
        #print(1/np.sqrt(nShots))
        #print(np.allclose(corrPOVM.get_POVM(),noisy_POVM.get_POVM(),atol=1/np.sqrt(nShots)))
        assert np.allclose(corrPOVM.get_POVM(),noisy_POVM.get_POVM(),atol=1/np.sqrt(nShots))
    
    end = time.time()
    print(f'Total dt runtime: {end - start}')


def measurement_test():
    np.random.seed(0)
    n_qubits=1
    start = time.time()
    n_states=10
    rho=np.array([sf.generate_random_pure_state(1) for _ in range(n_states)])#np.array([[1/2,1/2],[1/2,1/2]],dtype=complex)
    n_shots=10**6
    povm_mesh=POVM.generate_Pauli_POVM(n_qubits)
    outcome=np.zeros((3,10))
    
    for i in range (3):
        for j in range(n_states):
            out_temp=mf.measurement(n_shots,povm_mesh[i],rho[j],False,{})
        
            outcome[i,j]=np.sum(out_temp)/n_shots
    print(outcome)
    end = time.time()
    
    assert np.array_equal(outcome,np.array([[0.086265, 0.100177, 0.681203, 0.454587, 0.564957, 0.084083, 0.102104, 0.167757,
  0.735205, 0.185308],
 [0.517672, 0.69213,  0.281892, 0.463051 ,0.199737, 0.423756, 0.611773 ,0.81554,
  0.833552, 0.502384],
 [0.220338, 0.270147, 0.088223, 0.003455, 0.10485,  0.767328, 0.218644, 0.699056,
  0.788635, 0.111963]]))
    #print(outcome)
    print(f'Total measurment test runtime: {end - start}')
    return 1


def two_qubit_test():
    n_qubits=2
    
    shots=50

    calibration_states=sf.get_cailibration_states(n_qubits)
    POVM_list=POVM.generate_Pauli_POVM(n_qubits)
    #for state in calibration_states:
    nameList=["X+","X-","Y+","Y-","Z+","Z-"]
    full_name_list=[(a,b) for a in nameList for b in nameList]
    #for i in range(len(POVM_list)):
        #print(full_name_list[i])
        #outcomes=mf.simulated_measurement(shots,POVM_list[8],calibration_states[i])
        #print(POVM_list[i].get_angles())
        #print(outcomes)
    return 1

def noise_test():
    # Strenghts: 
    # 1: 0.05
    # 2: 0.2
    # 3: 0.01
    # 4: np.pi/5
    np.random.seed(0)
    POVMset=np.array([[[1,0],[0,0]],[[0,0],[0,1]]],dtype=complex)
    ini_POVM=POVM.z_axis_POVM(1)

    p_1=0.05
    p_2=0.2
    sol_1=(1-p_1)*POVMset[0] + p_1/2*np.eye(2)
    sol_2=(1-p_2)*POVMset[0] + p_2/2*np.eye(2)
    sol_3=np.array([[1.  +0.j, 0.  +0.j], [0.  +0.j, 0.01+0.j]])
    sol_4=np.array([[0.9045085+0.j,         0.       +0.29389263j],[0.       -0.29389263j, 0.0954915+0.j        ]])
    sol=np.array([sol_1,sol_2,sol_3,sol_4])
    for i in range(4):
        noise_POVM=POVM.generate_noisy_POVM(ini_POVM,i+1)
        #print(sol[i])
        #print(noise_POVM.get_POVM()[0])
        assert np.allclose(noise_POVM.get_POVM()[0],sol[i])
    
    POVMset=1/2*np.array([[[1,-1j],[1j,1]],[[1,1j],[-1j,1]]],dtype=complex)
    ini_POVM=POVM(POVMset)

    sol_1=(1-p_1)*POVMset[0] + p_1/2*np.eye(2)
    sol_2=(1-p_2)*POVMset[0] + p_2/2*np.eye(2)
    sol_3=np.array([[0.5+0.j,         0. -0.49749372j],[0. +0.49749372j, 0.5+0.j        ]])
    sol_4=np.array([[0.79389263+0.j,        0.        -0.4045085j],[0.        +0.4045085j, 0.20610737+0.j       ]])
    sol=np.array([sol_1,sol_2,sol_3,sol_4])
    for i in range(4):
        noise_POVM=POVM.generate_noisy_POVM(ini_POVM,i+1)
        #print(sol[i])
        #print(noise_POVM.get_POVM()[0])
        assert np.allclose(noise_POVM.get_POVM()[0],sol[i])
    #a=sf.Pauli_expectation_value(sol_4)
    #print(a)
if __name__=="__main__":
    main()
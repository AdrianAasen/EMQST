import numpy as np
from scipy.stats import unitary_group
import qutip as qt
from joblib import Parallel, delayed
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import sqrtm
import scipy



def function(Expdict):
    print(Expdict)

class NewClass:
    Noisemap=1

    def __init__(self,nQubtis=9):
        self.nQubits=nQubtis
    def noise(self):
        return self.Noisemap

def main():


    def key_dec(function):
        def wrapper():
            func=function()
            return function().keys()
        return wrapper
    @key_dec   
    def mesh():
        return {"Val": 10}

    #print(mesh())
    #x=NewClass()
    #print(x.noise())

    angles=np.array([[np.pi/2,np.pi/2],[np.pi/2,0]])
    rhos=get_density_matrix_from_angles(angles)
    #print(rhos)
    #rho=1/2*np.array([[1,1],[1,1]])
    for i in range(len(angles)):
        rho=get_density_matrix_from_angles(np.array([angles[i]]))
        #print(rho)
        #print(get_angles_from_density_matrix_single_qubit(rho))
    
    a=POVM.POVM_from_angles(angles)
    POVMList=a.POVM_list
    angles2=a.angle_representation
    #print(POVMList)
    #print(angles2)

    b=POVM(POVMList,angle_representation=angles2)
    #print(b.POVM_list==POVMList)
    #print(b.angle_representation)
    rho2=get_density_matrix_from_angles(angles)
    #print(rho2)
    rho=np.array([[1,0],[0,0]],dtype=complex)
    qbitrho2=np.kron(rho,rho)
    #outcomes=simulated_measurement(nShots,a,qbitrho2)




    #randomPOVM=generate_random_POVM(3)
    #print(randomPOVM)
    indexList=np.array([[0,0,3,1,1,0,1],[1,2,3,1,2,0,1]])
    index_list=np.arange(4)
    #print(index_list)
    #index_list=np.zeros((2**n_qubits,len(calibration_states)))
    index_counts=np.zeros((2,2**2))
    for i in range (len(indexList)):
        for j in range(len(index_list)):
            index_counts[i,j] = np.count_nonzero(indexList[i] == index_list[j])
    #print(index_counts)

    #eigV=np.array([1,2,3,4])
    #D =np.diag(1/np.sqrt(eigV))
    #print(D)
    
    POVMset=1/2*np.array([[[1,-1j],[1j,1]],[[1,1j],[-1j,1]]],dtype=complex)#np.array([[[1,0],[0,0]],[[0,0],[0,1]]],dtype=complex)#
    iniPOVM=POVM(POVMset,np.array([[[0,0],[np.pi,0]]]))
    #print(iniPOVM.get_POVM())
    #print(POVMset)
    #nShots=100
    expDict={}
    calibration_angles=np.array([[[np.pi/2,0]],[[np.pi/2,np.pi]],
                        [[np.pi/2,np.pi/2]],[[np.pi/2,3*np.pi/2]],
                        [[0,0]],[[np.pi,0]]])
    calibration_states=np.array([get_density_matrix_from_angles(angle) for angle in calibration_angles])

    

    


    # nShots=10**6
    # for i in range(1):
    #    corrPOVM=deviceTomography(1, nShots, noisy_POVM,calibration_states,expDict)
    #    print(noisy_POVM.get_POVM())
    #    print(f'Distance between reconstructed and noisy POVM: {POVM_distance(corrPOVM.get_POVM(),noisy_POVM.get_POVM())}')

    return 1





if __name__=="__main__":
    main()



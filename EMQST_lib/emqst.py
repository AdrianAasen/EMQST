import numpy as np
import os
import uuid
import matplotlib.pyplot as plt
from datetime import datetime
import time
from scipy.optimize import curve_fit

from EMQST_lib import support_functions as sf
from EMQST_lib import measurement_functions as mf
from EMQST_lib import dt
from EMQST_lib.qst import QST
from EMQST_lib.povm import POVM


def emqst(n_qubits,n_QST_shots_each,n_calibraion_shots_each,true_state_list, calibration_states=np.array([]),bool_exp_measurements=False,exp_dictionary={},n_cores=1,noise_mode=0):
    """
    Performs a complete cycle of noise corrected POVM with random sampling of states.
    Takes in experimental parameters. To be passed to both POVM calibration and QST. 
    Returns estimator dictionary.

    n_QST_shots_each        # shots in QST reconstruction for each POVM used. 
    n_calibraion_shots_each # shots for each calibration measurement.
    trueAngleList           List of random true states to average over.
    POVMCalibrationAngles   Angles used for calibrating the POVM, should be 3 MUBs.

    returns a list of mean corrected infidelities [len(TrueAngleList) x n_QST_shots_total],
                    list of uncorrected infidelities,
                    complex list of corrected_rho_estm [len(TrueAngleList) x 2 x 2 ]
    """

    

    # Check if restuls exist:
    check_path='results'
    path_exists=os.path.exists(check_path)
    if not path_exists:
        print("Created results dictionary.")
        os.makedirs('results')



    # Generate new dictionary for current run
    now=datetime.now()
    now_string = now.strftime("%Y-%m-%d_%H-%M-%S_")
    dir_name= now_string+str(uuid.uuid4())


    data_path=f'results/{dir_name}'
    os.mkdir(data_path)

    with open(f'{data_path}/experimental_settings.npy','wb') as f:
        np.save(f,exp_dictionary)

    if len(calibration_states)==0:
        calibration_states=sf.get_cailibration_states(n_qubits)
    
    POVM_list=POVM.generate_Pauli_POVM(n_qubits)

    print(f'----------------------------')
    print(f'Error corrected BME.\n{n_calibraion_shots_each*len(calibration_states)} POVM calibration shots.\n{n_QST_shots_each*len(POVM_list)} QST shots.\n{len(true_state_list)} QST averages.')
    print(f'----------------------------')


    # If experimental measurements are set, do not apply noise methods.
    if bool_exp_measurements:
        noise_mode=0
        print("Noise mode is disabled as experimental measurements are performed.")

    if noise_mode:
        noisy_POVM_list=np.array([POVM.generate_noisy_POVM(povm,noise_mode) for povm in POVM_list])
        print(f'Synthetic noise mode {noise_mode}.')
    else:
        noisy_POVM_list=POVM_list
        print("No synthetic noise.")
    dt_start=time.time()
    reconstructed_POVM_list = np.array([dt.device_tomography(n_qubits,n_calibraion_shots_each,noisy_POVM_list[i],calibration_states,bool_exp_measurements,exp_dictionary,initial_guess_POVM=POVM_list[i]) for i in range(len(noisy_POVM_list))])
    dt_end = time.time()
    print(f'Runtime of DT reconstruction {dt_end - dt_start}')
    DT_settings={
        "n_qubits": n_qubits,
        "calibration_states": calibration_states,
        "n_calibration_shots": n_calibraion_shots_each,
        "initial_POVM": POVM_list,
        "reconstructed_POVM_list": reconstructed_POVM_list,
        "bool_exp_meaurements": bool_exp_measurements,
        "noise_mode": noise_mode,
        "noisy_POVM_list" : noisy_POVM_list
    }

    with open(f'{data_path}/DT_settings.npy','wb') as f:
        np.save(f,DT_settings)

    

    for i in range (3):
        print(f'Distance between reconstructed and noisy POVM: {sf.POVM_distance(reconstructed_POVM_list[i].get_POVM(),noisy_POVM_list[i].get_POVM())}')

    print("POVM calibration complete.\n----------------------------")
    
    qst=QST(POVM_list,true_state_list,n_QST_shots_each,n_qubits,bool_exp_measurements,exp_dictionary,n_cores=n_cores,noise_corrected_POVM_list=reconstructed_POVM_list)
    qst.generate_data(override_POVM_list=noisy_POVM_list)
    # Save data settings
    qst.save_QST_settings(data_path,noise_mode)
    print("Generated data.")

    print("Start corrected QST.")
    qst.perform_BME(use_corrected_POVMs=True)
    
    corrected_infidelity=qst.get_infidelity()
    corrected_rho_estm=qst.get_rho_estm()
    print("Corrected QST complete.\n----------------------------")
    
    
    
    # Run comparative BME with uncorrected POVMs
    print("Start uncorrected QST.")
    
    qst.perform_BME()
    uncorrected_infidelity=qst.get_infidelity()
    uncorrected_rho_estm=qst.get_rho_estm()
    print("Uncorrected QST complete.\n----------------------------") 

    n_averages=len(true_state_list)
    sample_step=np.arange(len(uncorrected_infidelity[0]))
    corrected_average=np.sum(corrected_infidelity,axis=0)/n_averages
    uncorrected_average=np.sum(uncorrected_infidelity,axis=0)/n_averages

    with open(f'{data_path}/QST_results.npy','wb') as f:
        np.save(f,corrected_infidelity )
        np.save(f,uncorrected_infidelity)
        np.save(f,corrected_rho_estm)
        np.save(f,uncorrected_rho_estm)

    
    # Generate plots if not run on a cluster.
    if n_cores<10:
        cutoff=10
        popt_corr,pcov_corr=curve_fit(sf.power_law,sample_step[1000:],corrected_average[1000:],p0=np.array([1,-0.5]))
        corr_fit=sf.power_law(sample_step[cutoff:],popt_corr[0],popt_corr[1])
        popt_uncorr,pcov_uncorr=curve_fit(sf.power_law,sample_step[1000:],uncorrected_average[1000:],p0=np.array([1,-0.5]))
        uncorr_fit=sf.power_law(sample_step[cutoff:],popt_uncorr[0],popt_uncorr[1])

        plt.figure(figsize=(8,6))
        plt.plot(sample_step[cutoff:],corrected_average[cutoff:],'r', label="Corrected")
        plt.plot(sample_step[cutoff:],uncorrected_average[cutoff:],'b',label="Uncorrected")
        plt.plot(sample_step[cutoff:],corr_fit,'r--',label=rf'Fit, $N^a, a={"%.2f" % popt_corr[1]}$')
        plt.plot(sample_step[cutoff:],uncorr_fit,'b--',label=rf'Fit, $N^a, a={"%.2f" % popt_uncorr[1]}$')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlim(100,len(sample_step))
        #plt.ylim(10**(-5),10**(-0))
        plt.ylabel('Mean Infidelity')
        plt.xlabel('Number of shots')
        plt.legend(loc="lower left",prop={'size': 16})
        plt.tight_layout

        plt.savefig(f'{data_path}/Averaged_infidelities.png')
        plt.savefig('latest_run.png')


    
    return corrected_infidelity, uncorrected_infidelity, corrected_rho_estm


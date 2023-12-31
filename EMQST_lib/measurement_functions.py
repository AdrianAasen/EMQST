import numpy as np
from EMQST_lib import support_functions as sf



def measurement(n_shots,POVM,rho,bool_exp_measurements,exp_dictionary):
    """
    Measurment settings and selects either experimental or simulated measurements. 
    For experimental measurements some settings are converted to angle arrays. 
    """
    if bool_exp_measurements:
        outcome_index=exp_dictionary["measurement_function"](n_shots,POVM.get_angles(),sf.get_angles_from_density_matrix_single_qubit(rho),exp_dictionary)
    else:
        outcome_index=simulated_measurement(n_shots,POVM,rho)
    return outcome_index


def simulated_measurement(n_shots,POVM,rho):

    """
    Takes in number of shots required from a single POVM on a single quantum states.
    Returns and outcome_index vector where the index corresponds the the POVM that occured.
    """

    # Find probabilites for different outcomes
    histogram=POVM.get_histogram(rho)
    #print(histogram)
    cumulative_sum=np.cumsum(histogram)

    # Sample outcomes 
    r=np.random.random(n_shots)

    # Return index list of outcomes 
    return np.searchsorted(cumulative_sum,r)

import numpy as np

def dict_fault_ratio_fault(detected_issues):
    """Return a dictionary containing the ratio of false detected faults for each faulty scenario
    
    
   detected_issues: list of lenght number of faulty scenario.
   Each element of the list contains an array of size (n_simu,len_sample)
   The value of an element (i,j) of the array corresponds to number of features that found that for the simulation i,
   the observation j is an outlier.
    """
    dict_fault={}

    for fault_item in range(len(detected_issues)):
        aaa=detected_issues[fault_item]
        dnr_compute=0
        for line in range(len(aaa)):
            dnr_compute=dnr_compute+np.sum(aaa[line,160:]!=0)
        dnr_compute=dnr_compute/((960-160)*len(aaa))
        dict_fault['Fault_found_number_'+str(fault_item+1)]=dnr_compute
    return(dict_fault)


def dict_fault_ratio_false_faults_found(detected_issues):
    """Return a dictionary containing the ratio of false detected faults for each faulty scenario
    
    
   detected_issues: list of lenght number of faulty scenario.
   Each element of the list contains an array of size (n_simu,len_sample)
   The value of an element (i,j) of the array corresponds to number of features that found that for the simulation i,
   the observation j is an outlier.
    """
    
    dict_fault={}

    for fault_item in range(len(detected_issues)):
        aaa=detected_issues[fault_item]
        dnr_compute=0
        for line in range(len(aaa)):
            dnr_compute=dnr_compute+np.sum(aaa[line,0:160]!=0)
        dnr_compute=dnr_compute/((160)*len(aaa))
        dict_fault['Fault_found_number_'+str(fault_item+1)]=dnr_compute
    return(dict_fault)
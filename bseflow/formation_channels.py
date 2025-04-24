import numpy as np 
from bseflow.file_processing import multiprocess_files
from bseflow.utils import in1d
import collections

def mask_DCO_type(file, selected_seeds=None):
    """Create arrays labelled '1' for specific DCO type.

    Parameters
    ----------
    file : `h5 file`
    selected_seeds : `int/array`
        List of seeds that correspond to a binary and each instance
        corresponds to a moment of mass transfer.
    
    Returns
    -------
    BNS_mask : `array`
        Array same shape as ``seeds`` with each index flagged if it becomes a BNS or not.
    BBH_mask : `array`
        Array same shape as ``seeds`` with each index flagged if it becomes a BBH or not.
    NSBH_mask : `array`
        Array same shape as ``seeds`` with each index flagged if it becomes a NSBH or not.    
    """
    
    files_and_fields = [(file, 'systems', ['SEED'], {}),
            (file, 'doubleCompactObjects', ['M1', 'M2'], {})]

    (sys_seeds, _), \
    (dco_seeds, M1, M2) = multiprocess_files(files_and_fields, return_df=False, num_processes=2, preview=False, selected_seeds=selected_seeds)

    M1isNS = (M1<=2.5)
    M2isNS = (M2<=2.5)

    BNS_DCO_mask = ( (M1isNS==1) & (M1isNS==1))
    BNS_mask = in1d(sys_seeds, dco_seeds[BNS_DCO_mask])

    BBH_DCO_mask = ( (M1isNS==0) & (M1isNS==0))
    BBH_mask = in1d(sys_seeds, dco_seeds[BBH_DCO_mask])

    NSBH_DCO_mask = ( (M1isNS==1) & (M2isNS==0)) | ((M1isNS==0) & (M2isNS==1) ) 
    NSBH_mask = in1d(sys_seeds, dco_seeds[NSBH_DCO_mask])

    return BNS_mask, BBH_mask, NSBH_mask


def mask_mass_transfer_episodes(file, selected_seeds=None):
    """Create an array labelled '1' for mass transfer moments before SN1.

    Create an array indicating if mass transfer is pre SN1.
    '1' indicates pre SN1. '0' indicates neither. Will only flag systems
    that undergo at least one supernovae.

    Parameters
    ----------
    file : `h5 file`
    selected_seeds : `int/array`
        List of seeds that correspond to a binary and each instance
        corresponds to a moment of mass transfer.

    Returns
    -------
    pre_sn1_mask : `array`
        Array same shape as ``seeds`` with each index flagged if MT pre SN1.
    post_sn1_mask: `array`
        Array same shape as ``seeds`` with each index flagged if MT post SN1.
    """

    files_and_fields = [(file, 'RLOF', ['type1Prev', 'type2Prev', 'type1', 'type2'], {}),
                        (file, 'supernovae', ['randomSeed'], {}),
                        (file, 'commonEnvelopes', ['stellarType1', 'finalStellarType1', 'stellarType2', 'finalStellarType2'], {})]
    
    (rlof_seeds, prev_stellar_type_1, prev_stellar_type_2, stellar_type_1, stellar_type_2), \
    (sn_seeds, _), \
    (ce_seeds, ce_type1_prev, ce_type1, ce_type2_prev, ce_type2) = multiprocess_files(files_and_fields, return_df=False, num_processes=2, preview=False, selected_seeds=selected_seeds)

    SN_RLOF_mask = in1d(rlof_seeds, sn_seeds)
    SN_CE_mask = in1d(ce_seeds, sn_seeds)

    # not a significant mass transfer if prevtype1 < 13 and type1 >=13
    ## edge case, rlof initiated but SN happens during -- disregard for now

    invalid_mask = ( ( (prev_stellar_type_1 < 13) & (stellar_type_1 >= 13) |
                    (prev_stellar_type_2 < 13) & (stellar_type_2 >= 13) ) & (SN_RLOF_mask==1) )

    pre_sn1_mask = ( (stellar_type_1 < 13) & 
                           (stellar_type_2 < 13) & (invalid_mask==0) )
    
    pre_sn1_ce_mask = ( (ce_type1_prev < 13) & 
                           (ce_type2_prev < 13))
    
    post_sn1_mask = (  ( ((prev_stellar_type_1 >= 13) & (prev_stellar_type_2 < 13)) | ((prev_stellar_type_1 < 13) & (prev_stellar_type_2 >= 13)) )
                     & (invalid_mask==0) & (SN_RLOF_mask==1) )
    
    post_sn1_ce_mask = (  ( ((ce_type1_prev >= 13) & (ce_type2_prev < 13)) | ((ce_type1_prev < 13) & (ce_type2_prev >= 13)) )
                     & (SN_CE_mask==1) )
    
    return pre_sn1_mask, post_sn1_mask, invalid_mask, pre_sn1_ce_mask, post_sn1_ce_mask


def identify_formation_channels(file, selected_seeds=None):

    files_and_fields = [
    (file, 'systems', ['weight'], {}),
    (file, 'RLOF', ['flagCEE', 'flagRLOF1', 'flagRLOF2', 'type1Prev', 'type2Prev'], 
        {}),
    (file, 'commonEnvelopes', ['randomSeed'], {})  
    ]

    (sys_seeds, weight), \
    (rlof_seeds, flagCEE, rlof_primary, rlof_secondary, type1, type2), \
    (ce_seeds, _) = multiprocess_files(files_and_fields, return_df=False, num_processes=2, preview=False, selected_seeds=selected_seeds)
    
    pre_sn_mask, post_sn_mask, invalid_rlof_mask, pre_sn_ce_mask, post_sn_ce_mask = mask_mass_transfer_episodes(file, selected_seeds=selected_seeds)

    valid_rlof_seeds = rlof_seeds[~invalid_rlof_mask]

    RLOF_mask = in1d(sys_seeds, valid_rlof_seeds)  # systems that experience MT that aren't weird

    mask_dict = {}

    # No mass transfer
    no_MT_mask = ~RLOF_mask
    mask_dict['no_MT'] = no_MT_mask

    no_MT_pre_SN_mask = ( ( (~in1d(sys_seeds, rlof_seeds[pre_sn_mask]) ) & (~in1d(sys_seeds, ce_seeds[pre_sn_ce_mask]))) | ~RLOF_mask )
    no_MT_post_SN_mask = ( ( (~in1d(sys_seeds, rlof_seeds[post_sn_mask]) ) & (~in1d(sys_seeds, ce_seeds[post_sn_ce_mask]))) | ~RLOF_mask )
    mask_dict['no_MT_pre_SN'] = no_MT_pre_SN_mask
    mask_dict['no_MT_post_SN'] = no_MT_post_SN_mask

    # CEE masks
    # CE MT pre SN
    CE_pre_SN_seeds = np.concatenate( (rlof_seeds[(pre_sn_mask==1) & (flagCEE==1)], ce_seeds[pre_sn_ce_mask]) )
    CE_pre_SN_RLOF_mask = in1d(rlof_seeds, CE_pre_SN_seeds)
    CE_pre_SN_mask = in1d(sys_seeds, CE_pre_SN_seeds)
    mask_dict['CE_pre_SN'] = CE_pre_SN_mask
    only_stable = ~in1d(rlof_seeds, ce_seeds)

    # SMT pre SN
    OS_pre_SN_seeds = rlof_seeds[(CE_pre_SN_RLOF_mask==0) & (pre_sn_mask==1)]
    OS_pre_SN_mask = in1d(sys_seeds, OS_pre_SN_seeds)
    mask_dict['OS_pre_SN'] = OS_pre_SN_mask

    # CE MT post SN
    CE_post_SN_seeds = np.concatenate( (rlof_seeds[(post_sn_mask==1) & (flagCEE==1)], ce_seeds[post_sn_ce_mask]) )
    CE_post_SN_RLOF_mask = in1d(rlof_seeds, CE_post_SN_seeds)
    CE_post_SN_mask = in1d(sys_seeds, CE_post_SN_seeds)
    mask_dict['CE_post_SN'] = CE_post_SN_mask

    # SMT post SN
    OS_post_SN_seeds = rlof_seeds[(CE_post_SN_RLOF_mask==0) & (post_sn_mask==1)]
    OS_post_SN_mask = in1d(sys_seeds, OS_post_SN_seeds)
    mask_dict['OS_post_SN'] = OS_post_SN_mask

    # Classic
    # Stable RLOF from primary (post-MS, unstripped) onto MS
    classic_or_OS_MT1 = ( (pre_sn_mask==1) & (rlof_primary==1) & (CE_pre_SN_RLOF_mask==0)
                         & (type1<10) & (type2<10) )
                #          & (type1>1) & (type1<7)
                # & (type2<=1) ) # allow for classic to include stable MT 1 of all stellar types
    classic_MT1_mask = in1d(sys_seeds, rlof_seeds[classic_or_OS_MT1])
    mask_dict['classic_MT1'] = classic_MT1_mask
    
    # 2nd MT, unstripped secondary RLOF CE
    classic_MT2 = ( (post_sn_mask==1) & (rlof_secondary==1) & (CE_post_SN_RLOF_mask==1) & ((type1<10) | (type2<10)) )
    classic_MT2_seeds = np.concatenate((rlof_seeds[classic_MT2], ce_seeds[post_sn_ce_mask]))

    classic_seeds = np.intersect1d(rlof_seeds[classic_or_OS_MT1], classic_MT2_seeds)
    classic_mask = in1d(sys_seeds, classic_seeds)
    mask_dict['classic'] = classic_mask

    # Only Stable
    OS_seeds = rlof_seeds[only_stable]
    OS_mask = in1d(sys_seeds, OS_seeds)
    mask_dict['OSMT'] = OS_mask

    mask_dict['OSMT1'] = (no_MT_pre_SN_mask==0) & (no_MT_post_SN_mask==0) & OS_mask   # SMT pre & post SN
    mask_dict['OSMT2'] = no_MT_post_SN_mask & OS_mask  # SMT pre SN
    mask_dict['OSMT3'] = no_MT_pre_SN_mask & OS_mask # SMT post SN

    # Single Core CEE
    # 1st transfer unstable, primary giant branch onto MS secondary
    single_core = ( (pre_sn_mask==1) & (rlof_primary==1) & (CE_pre_SN_RLOF_mask==1) &  ( (type1>2) & (type1<7) 
        &  (type2<=1) )  )  ## fix this, weird
    
    single_core_seeds = rlof_seeds[single_core]
    single_core_mask = in1d(sys_seeds, single_core_seeds)
    mask_dict['single_core'] = single_core_mask

    # Double Core CEE
    double_core = ( (pre_sn_mask==1) & (rlof_primary==1) & (CE_pre_SN_RLOF_mask==1) & ( (type1>2) & (type1<7) ) & ( (type2>2) & (type2<7) ) )
    double_core_seeds = rlof_seeds[double_core]
    double_core_mask = in1d(sys_seeds, double_core_seeds)
    mask_dict['double_core'] = double_core_mask

    # Classic Case A
    # 1st MT as classic case A, 2nd MT unstripped secondary stable RLOF
    classic_caseA_MT1 = ( (pre_sn_mask==1) & (CE_post_SN_RLOF_mask==0) & (type1==1) & (type2==1) )
    
    only_stable_caseA = (classic_caseA_MT1==1) & (only_stable==1)
    only_stable_caseA_mask = in1d(sys_seeds, rlof_seeds[only_stable_caseA])
    mask_dict['classic_caseA_MT1'] = only_stable_caseA_mask

    # if CE, overwrite
    classic_caseA = (classic_caseA_MT1==1) & (only_stable==0)
    classic_caseA_mask = in1d(sys_seeds, rlof_seeds[classic_caseA])
    mask_dict['classic_caseA'] = classic_caseA_mask

    channels=np.zeros_like(sys_seeds).astype(int)

    channels[no_MT_mask] = 7
    # channels[CE_pre_SN_mask] = 8
    # channels[OS_pre_SN_mask] = 9
    # channels[CE_post_SN_mask] = 10
    # channels[OS_post_SN_mask] = 11

    channels[classic_mask] = 1
    channels[OS_mask] = 2
    channels[single_core_mask] = 3
    channels[double_core_mask] = 4
    channels[classic_caseA_mask] = 5
    channels[only_stable_caseA_mask] = 6

    return channels, mask_dict
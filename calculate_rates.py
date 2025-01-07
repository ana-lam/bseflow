import numpy as np
from file_processing import multiprocess_files
import pandas as pd
from formation_channels import mask_mass_transfer_episodes, mask_DCO_type
from utils import in1d
from total_mass_evolved_per_Z import totalMassEvolvedPerZ


def clean_buggy_systems(file, selected_seeds=None):

    files_and_fields = [
    (file, 'systems', ['mass1', 'stellar_merger', 'disbound', 'weight'], {}),
    (file, 'commonEnvelopes', ['stellarType1', 'stellarType2', 'stellarMerger'], {}),
    (file, 'RLOF', ['radius1', 'radius2'], {}),
    (file, 'supernovae', [], {}),
    (file, 'formationChannels', ['stellar_type_K1', 'stellar_type_K2'], {})  
    ]

    (sys_seeds, ZAMSmass1, stellar_merger, disbound, weight), \
    (ce_seeds, stellarType1, stellarType2, ce_stellar_merger), \
    (rlof_seeds, radius1, radius2),  \
    (sn_seeds), \
    (f_seeds, stellar_type_K1, stellar_type_K2) = multiprocess_files(files_and_fields, selected_seeds=selected_seeds)

    # clean systems (negative radii)
    buggy_seeds = rlof_seeds[(radius1<0) | (radius2<0)]

    # clean more systems (don't stellar merger, don't disbound, and remain stellar type<10)
    pre_sn_mask, post_sn_mask, invalid_rlof_mask, pre_sn_ce_mask, post_sn_ce_mask = mask_mass_transfer_episodes(file, selected_seeds=selected_seeds)
    RLOF_mask = in1d(sys_seeds, rlof_seeds[~invalid_rlof_mask])


    non_dco_f_seeds = f_seeds[((stellar_type_K1<10) | (stellar_type_K2<10))]
    non_dco_sys_mask = in1d(sys_seeds, non_dco_f_seeds)

    unique_SN_seeds, sn_counts = np.unique(sn_seeds[0], return_counts=True)
    sys_SN2_mask = in1d(sys_seeds, unique_SN_seeds[sn_counts>1])
    merger_RLOF_seeds = sys_seeds[(RLOF_mask) & (stellar_merger==1)]
    merger_CE_seeds = ce_seeds[(ce_stellar_merger==1)]
    merger_seeds = np.concatenate((merger_RLOF_seeds, merger_CE_seeds))

    merger_mask = in1d(sys_seeds, merger_seeds)

    disregard = sys_seeds[(merger_mask==0) & (sys_SN2_mask==0)]
    sys_disregard = in1d(sys_seeds, disregard)

    more_buggy_seeds = sys_seeds[(stellar_merger==0) & (sys_disregard==1) & (disbound==0) & (non_dco_sys_mask==1)] # assume these become WD, so add to WD factor
    WD_factor = np.sum(weight[in1d(sys_seeds, more_buggy_seeds)])/np.sum(weight)

    _, totalMass = totalMassEvolvedPerZ(pathCOMPASh5=file, Mlower=min(ZAMSmass1), Mupper=max(ZAMSmass1))
    totalMass = np.sum(totalMass)
    WD_rate = np.divide(np.sum(weight[in1d(sys_seeds, more_buggy_seeds)]), totalMass)


    buggy_seeds = np.concatenate((buggy_seeds, more_buggy_seeds))


    # clean more buggy systems, stellar type > 9 but no SN (weird accretion stuff)
    ce_seeds_in_SN = in1d(ce_seeds, sn_seeds[0])
    buggy_ce_seeds = ce_seeds[(ce_seeds_in_SN==0) & ( (stellarType1>9) | (stellarType2>9) )]
    
    # invalid RLOF seeds, weird MT & SN time stepping issue seeds
    invalid_RLOF_seeds = rlof_seeds[invalid_rlof_mask]

    buggy_seeds = np.concatenate((buggy_seeds, buggy_ce_seeds, invalid_RLOF_seeds))

    sys_mask = (~in1d(sys_seeds, buggy_seeds))

    return sys_seeds[sys_mask], WD_factor, WD_rate

def specify_metallicity(Z, file, selected_seeds=None):

    files_and_fields = [
    (file, 'systems', ['Metallicity1'], {})
    ]

    (sys_seeds, stellar_merger, disbound, weight) = multiprocess_files(files_and_fields, selected_seeds=selected_seeds)

    return Z_seeds


def calculate_rate(data, systems, weights, total_weight, metallicities, unique_Z, total_mass, 
                   CEE=None, condition=None, addtnl_ce_seeds = None, rel_rate=None, formation_channel=None, WD_mask=None):
    """Returns a dict with seeds, count, and rate for specific evolution events
    
    Parameters
    ----------
    data : `np.array`
        seed data we are using in calculation
    systems : `np.array`
        seeds of all systems
    weights : `np.array`
        specific array of weights
    CEE : `np.array`, optional
        mask for CEE evolution
    condition : `np.array`, optional
        boolean mask to filter data
    rel_rate : `np.array`, optional
        seeds to compute relative rate
    formation_channel : `np.array`, optional
        boolean mask for formation channel
    
    Returns
    -------
    result_dict : `dict`
        dictionary of seeds, count, rate, rel_rate, and cee_rate for specific evolution event
    """
    
    # standard rate
    def compute_rate(mask, total_weight, fc=None, addtnl_ce_seeds=None, rel_rate=None, WD_mask=None):
        
        masked_data = data[mask]
        if addtnl_ce_seeds is not None:
            masked_data = np.concatenate((masked_data, addtnl_ce_seeds))
        if fc is not None:
            masked_data = masked_data[in1d(masked_data, fc)] 
        if WD_mask is not None:
            masked_data = masked_data[~in1d(masked_data, WD_mask)]
        mask = in1d(systems, masked_data)
        total_count = np.sum(weights[mask])
        rate = total_count / total_weight

        if rel_rate is not None:
            rel_rate_seeds = np.concatenate([r['seeds'] for r in rel_rate if 'seeds' in r]) if isinstance(rel_rate, list) else rel_rate['seeds']
            rel_rate = total_count / np.sum(weights[in1d(systems, rel_rate_seeds)])
        else:
            rel_rate = rate
        return systems[mask], total_count, rate, rel_rate
    
    # astrophysical rate
    def compute_rates_per_mass(seeds):

        seed_mask = in1d(systems, seeds)
        formation_rate = np.divide(np.sum(weights[seed_mask]), total_mass)
        
        formation_rates = np.array([np.sum(weights[seed_mask & (metallicities == Z)]) for Z in unique_Z])

        # formation_rates = np.zeros(len(unique_Z))

        # seed_metallicities = metallicities[seed_mask]
        # seed_weights = weights[seed_mask]
        
        # for index, Z in enumerate(unique_Z):
        #     if Z in seed_metallicities:
        #         maskZ = (seed_metallicities==Z)
        #         formation_rates[index] = np.sum(seed_weights[maskZ])
        #     else:
        #         formation_rates[index] = 0
        
        # formation_rates = None
        ## formation rates per metallicities
        formation_rates = np.divide(formation_rates, total_mass)
        
        return formation_rate, formation_rates
    
    mask = condition if condition is not None else np.ones_like(data, dtype=bool)
    # if formation_channel is not None:
    #     mask &= formation_channel

    ce_seeds, smt_seeds, ce_rate, smt_rate, ce_rel_rate, smt_rel_rate = None, None, None, None, None, None
    ce_rate_per_mass, smt_rate_per_mass, ce_rates_per_mass, smt_rates_per_mass = None, None, None, None
    
    if CEE is not None:
        ce_seeds, ce_total_count, ce_rate, ce_rel_rate = compute_rate(mask & (in1d(data, CEE)), total_weight,  fc=formation_channel, addtnl_ce_seeds=addtnl_ce_seeds[in1d(addtnl_ce_seeds, CEE)] if addtnl_ce_seeds is not None else addtnl_ce_seeds, rel_rate=rel_rate)
        ce_rate_per_mass, ce_rates_per_mass = compute_rates_per_mass(ce_seeds)
        smt_seeds, smt_total_count, smt_rate, smt_rel_rate = compute_rate(mask & (~in1d(data, CEE)), total_weight,  fc=formation_channel, addtnl_ce_seeds=addtnl_ce_seeds[~in1d(addtnl_ce_seeds, CEE)] if addtnl_ce_seeds is not None else addtnl_ce_seeds, rel_rate=rel_rate)
        smt_rate_per_mass, smt_rates_per_mass = compute_rates_per_mass(smt_seeds)

    unique_seeds, total_count, rate, rel_rate = compute_rate(mask, total_weight, fc=formation_channel, addtnl_ce_seeds=addtnl_ce_seeds, rel_rate=rel_rate, WD_mask=WD_mask)
    rate_per_mass, rates_per_mass = compute_rates_per_mass(unique_seeds)

    result_dict = {
        'seeds': unique_seeds.astype(int), 
        'count': total_count, 
        'rate': rate, 
        'rel_rate': rel_rate, 
        'cee_rate': [ce_rate, smt_rate],
        'cee_rel_rate': [ce_rel_rate, smt_rel_rate],
        'ce_seeds': ce_seeds.astype(int) if ce_seeds is not None else ce_seeds,
        'smt_seeds': smt_seeds.astype(int) if smt_seeds is not None else smt_seeds,
        'rate_per_mass' : rate_per_mass,
        'rates_per_mass' : rates_per_mass,
        'cee_rate_per_mass' : [ce_rate_per_mass, smt_rate_per_mass],
        'cee_rates_per_mass' : [ce_rates_per_mass, smt_rates_per_mass],
        'unique_Z' : unique_Z
        }

    return result_dict


def calculate_simulation_rates(file, CEE=False, white_dwarfs=False, PISN=False, weights=None, 
                               selected_seeds=None, formation_channel=None, formation_channel_2=None, optimistic_CE=False, additional_WD_factor=None):
    """
    Creates a dictionary of seeds, count, rates for critical events in BSE

    Parameters
    ----------
    file : `h5 file`
        BSE pop synth output
    weights : `np.array`
        specific weights to use in calculation
    CEE : `bool`
        split MT calculations into SMT and CEE 
    selected_seeds : `np.array`
        calculate for only selected seeds
    formation_channel : `np.array`
        mask for specific formation channel
    formation_channel_2 : `np.array`
        when masking formation channel by 1st and 2nd MT, this is for 2nd MT
    
    Returns
    -------
    rates_dict : `dict`
        dictionary of seeds, count, rate, rel_rate for critical events in BSE
    rates_df : `pd.DataFrame`
        data frame with rates for critical events in BSE
    """

    rates_dict = {} # dictionary to store calculations

    # define and pull specific fields we need to calculate rates
    files_and_fields = [
    (file, 'systems', ['weight', 'stellar_merger', 'mass1', 'mass2', 'Metallicity1', 'disbound'], {}),
     (file, 'RLOF', ['flagCEE', 'type1', 'type2', 'type1Prev', 'type2Prev', 'mass1', 'mass2', 'radius1', 'radius2'], 
        {}),
    (file, 'doubleCompactObjects', ['mergesInHubbleTimeFlag'],
        {}),
    (file, 'supernovae', ['Survived', 'previousStellarTypeSN', 'previousStellarTypeCompanion', 'flagPISN', 'flagPPISN'], {}),
    (file, 'commonEnvelopes', ['stellarMerger', 'stellarType1', 'stellarType2', 'finalStellarType1', 'stellarType2', 'optimisticCommonEnvelopeFlag'], {}),
    (file, 'formationChannels', ['stellar_type_K1', 'stellar_type_K2'], {})  
    ]

    (sys_seeds, weight, stellar_merger, ZAMSmass1, ZAMSmass2, metallicity1, disbound), \
    (rlof_seeds, flagCEE, type1, type2, type1Prev, type2Prev, RLOFmass1, RLOFmass2, radius1, radius2),  \
    (dco_seeds, mergesInHubbleTimeFlag), \
    (sn_seeds, survived, previousStellarTypeSN, previousStellarTypeCompanion, flagPISN, flagPPISN), \
    (ce_seeds, ce_stellar_merger, ce_type1, ce_type2, ce_type1_final, ce_type2_final, optimisticCEflag), \
    (f_seeds, stellar_type_K1, stellar_type_K2) = multiprocess_files(files_and_fields, selected_seeds=selected_seeds)

    # total mass evolved per Z for astrophysical rates
    _, totalMass = totalMassEvolvedPerZ(pathCOMPASh5=file, Mlower=min(ZAMSmass1), Mupper=max(ZAMSmass1))
    totalMass = np.sum(totalMass)

    # define weights if specified 
    weights = weights if weights is not None else weight
    total_weight = np.sum(weights)

    # pull unique metallicities
    unique_Z = np.unique(metallicity1)  

    sys_mask = None

    # clean out weird MT & SN time stepping issue seeds
    pre_sn_mask, post_sn_mask, invalid_rlof_mask, pre_sn_ce_mask, post_sn_ce_mask = mask_mass_transfer_episodes(file, selected_seeds=selected_seeds)

    # always clean out massless remnants (stellar type ==15)
    mr_seeds = f_seeds[(stellar_type_K1==15) | (stellar_type_K2==15)]
    mr_seeds_rlof = rlof_seeds[(type1==15) | (type2==15)]
    mr_seeds = np.concatenate((mr_seeds, mr_seeds_rlof))
    mr_mask = in1d(sys_seeds, mr_seeds)
    rates_dict['MR'] = calculate_rate(sys_seeds, sys_seeds, weights, total_weight, metallicity1, unique_Z, totalMass,
                                            condition = mr_mask)
    sys_mask = (~mr_mask) 

    # clean out white dwarfs if not specified else wise
    # white dwarf that goes supernova (NS)
    WD_SN_seeds = sn_seeds[((previousStellarTypeSN>9) & (previousStellarTypeSN<13)) | ((previousStellarTypeCompanion>9) & (previousStellarTypeCompanion<13))]
    pre_sn_mask, post_sn_mask, invalid_rlof_mask, pre_sn_ce_mask, post_sn_ce_mask = mask_mass_transfer_episodes(file, selected_seeds=selected_seeds)
    RLOF_mask = in1d(sys_seeds, rlof_seeds[~invalid_rlof_mask])
    # RLOF_SN_mask = in1d(rlof_seeds, sn_seeds)
    # merger_RLOF_seeds = sys_seeds[(RLOF_mask) & (stellar_merger==1)]
    # merger_RLOF_mask = in1d(rlof_seeds, merger_RLOF_seeds)
    # MT1_WD_seeds = rlof_seeds[( (pre_sn_mask==1) & (merger_RLOF_mask==0) & (RLOF_SN_mask==0) & 
    #                         ( ( ( ( (type1 > 9 ) & (type1 < 13) ) | ( (type2 > 9 ) & (type2 < 13) ) )
    #                         | ( ( (type1Prev > 9 ) & (type1Prev < 13) ) | ( (type2Prev > 9 ) & (type2Prev < 13) ) ) ) ) | 
    #                         ( (pre_sn_mask==1) & (merger_RLOF_mask==0) & (RLOF_SN_mask==0)) )]
    # MT2_WD_seeds = rlof_seeds[( (post_sn_mask==1) & (merger_RLOF_mask==0) &
    #                         ( ( ( (type1 > 9 ) & (type1 < 13) ) | ( (type2 > 9 ) & (type2 < 13) ) )
    #                         | ( ( (type1Prev > 9 ) & (type1Prev < 13) ) | ( (type2Prev > 9 ) & (type2Prev < 13) ) ) ) | 
    #                         ((post_sn_mask==1) & (RLOFmass2<1)) )]
    RLOF_WD_seeds = rlof_seeds[ ( ( ( (type1 > 9 ) & (type1 < 13) ) | ( (type2 > 9 ) & (type2 < 13) ) )
                            | ( ( (type1Prev > 9 ) & (type1Prev < 13) ) | ( (type2Prev > 9 ) & (type2Prev < 13) ) ) )]
    
    # let's discard white dwarfs (& stellar type 15 systems)
    k_type = f_seeds[ ( (stellar_type_K1>9) & (stellar_type_K1<13) ) | ( (stellar_type_K2>9) & (stellar_type_K2<13) )]
    # totalZAMSmass = ZAMSmass1 + ZAMSmass2
    # white_dwarf_seeds = np.concatenate((rlof_seeds[ ( ( ( (type1>9 ) & (type1<13) ) | ( (type2>9 ) & (type2<13) ) )
    #                                         | ( ( (type1Prev>9 ) & (type1Prev<13) ) | ( (type2Prev>9 ) & (type2Prev<13) ) ) )],
    #                                     sys_seeds[(ZAMSmass1<8)], MT1_WD_seeds, WD_SN_seeds, MT2_WD_seeds, k_type))
    white_dwarf_seeds = np.concatenate((RLOF_WD_seeds, WD_SN_seeds, k_type))

    white_dwarf_seeds = white_dwarf_seeds[~in1d(white_dwarf_seeds, dco_seeds)]
    wd_mask = in1d(sys_seeds, white_dwarf_seeds)

    if white_dwarfs is False:
        sys_mask = (sys_mask) & (~wd_mask)

        rates_dict['WD'] = calculate_rate(sys_seeds, sys_seeds, weights, total_weight, metallicity1, unique_Z, totalMass,
                                            condition = wd_mask & (~mr_mask)) 
        
        # save white dwarf count for drake factor
        wd_count = np.sum(weights[wd_mask])
        wd_factor = wd_count/total_weight

        wd_rate = wd_count/totalMass


        if additional_WD_factor:
            wd_factor += additional_WD_factor[0]
            wd_rate += additional_WD_factor[1]

    # clean out PISN systems if not specified elsewise
    if not PISN:
        # let's discard PISN systems
        PISN_seeds = np.concatenate((sn_seeds[flagPISN==1], sn_seeds[flagPPISN==1]))
        PISN_mask = in1d(sys_seeds, PISN_seeds)
        sys_mask = sys_mask & (~PISN_mask)
        rates_dict['PISN'] = calculate_rate(sys_seeds, sys_seeds, weights, total_weight, metallicity1, unique_Z, totalMass,
                                            condition = PISN_mask)

    if sys_mask is not None:
        sys_seeds, weights, stellar_merger, ZAMSmass1, metallicity1 = (
                sys_seeds[sys_mask],
                weights[sys_mask],
                stellar_merger[sys_mask],
                ZAMSmass1[sys_mask],
                metallicity1[sys_mask]
            )
        total_weight = np.sum(weights)

    # calculate rates for a specific formation channel if specified
    if formation_channel is not None:
        fc_seeds = sys_seeds[formation_channel[sys_mask]] if sys_mask is not None else sys_seeds[formation_channel]
    else:
        fc_seeds = None

    # assume HG donor stars who initiate CE do not survive unless specified as optimistic_CE

    optimistic_CE_pre_sn_CE_mask = (optimisticCEflag==1) & (ce_type1 < 13) & (ce_type2 < 13)
    optimistic_CE_post_sn_CE_mask = (optimisticCEflag==1) & ( (ce_type1>12) | (ce_type2>12))

    optimistic_CE_pre_sn_seeds = ce_seeds[optimistic_CE_pre_sn_CE_mask]
    optimistic_CE_post_sn_seeds = ce_seeds[optimistic_CE_post_sn_CE_mask]
    
    optimistic_CE_pre_sn_mask = in1d(rlof_seeds, optimistic_CE_pre_sn_seeds)
    optimistic_CE_post_sn_mask = in1d(rlof_seeds, optimistic_CE_post_sn_seeds)

    optimistic_CE_pre_sn_SN_mask = in1d(sn_seeds, optimistic_CE_pre_sn_seeds)
    optimistic_CE_post_sn_SN_mask = in1d(sn_seeds, optimistic_CE_post_sn_seeds)

    optimistic_CE_pre_sn_DCO_mask = in1d(dco_seeds, optimistic_CE_pre_sn_seeds)
    optimistic_CE_post_sn_DCO_mask = in1d(dco_seeds, optimistic_CE_post_sn_seeds) 

    # ZAMS
    rates_dict['ZAMS'] = calculate_rate(sys_seeds, sys_seeds, weights, total_weight, metallicity1, unique_Z, totalMass)
    # MT masks
    pre_sn_mask, post_sn_mask, invalid_rlof_mask, pre_sn_ce_mask, post_sn_ce_mask = mask_mass_transfer_episodes(file, selected_seeds=selected_seeds)
    RLOF_mask = in1d(sys_seeds, rlof_seeds[~invalid_rlof_mask])
    CEE_mask = in1d(sys_seeds, ce_seeds)

    # CE masks
    merger_pre_sn = ce_seeds[(ce_stellar_merger==1) & ( (ce_type1<13) | (ce_type2<13) )]
    merger_post_sn = ce_seeds[(ce_stellar_merger==1) & ( ((ce_type1>12) & (ce_type1<15)) | ((ce_type2>12) & (ce_type2<15)) )]
    RLOF_merger_pre_sn_mask = in1d(rlof_seeds, merger_pre_sn)
    RLOF_merger_post_sn_mask = in1d(rlof_seeds, merger_post_sn)

    if CEE:
        CE_pre_SN_seeds = rlof_seeds[(flagCEE==1) & (pre_sn_mask==1)]
        CE_pre_SN_seeds = np.concatenate((ce_seeds[pre_sn_ce_mask==1], CE_pre_SN_seeds))
        CE_post_SN_seeds = rlof_seeds[(flagCEE==1) & (post_sn_mask==1)]
        CE_post_SN_seeds = np.concatenate((ce_seeds[post_sn_ce_mask==1], CE_post_SN_seeds))
    else:
        CE_pre_SN_seeds = CE_pre_SN_mask = CE_post_SN_seeds = CE_post_SN_mask = None

    # merger MT mask
    merger_RLOF_seeds = sys_seeds[(RLOF_mask) & (stellar_merger==1)]
    merger_RLOF_mask = in1d(rlof_seeds, merger_RLOF_seeds)

    # SN masks
    unique_SN_seeds, sn_counts = np.unique(sn_seeds, return_counts=True) # systems that go SN
    SN_mask = in1d(sys_seeds, sn_seeds)
    SN_SN2_mask = in1d(sn_seeds, unique_SN_seeds[sn_counts>1])
    RLOF_SN_mask = in1d(rlof_seeds, sn_seeds)
    RLOF_SN_2_mask = in1d(rlof_seeds, sn_seeds[SN_SN2_mask])
    CE_SN_2_mask = in1d(ce_seeds, sn_seeds[SN_SN2_mask])
    pre_MT_seeds = np.concatenate((rlof_seeds[pre_sn_mask==1], ce_seeds[pre_sn_ce_mask==1]))
    preSN_RLOF_mask  = in1d(sn_seeds, pre_MT_seeds)
    RLOF_SN2_mask = in1d(rlof_seeds, unique_SN_seeds[sn_counts>1])
    post_MT_seeds = np.concatenate((rlof_seeds[post_sn_mask==1], ce_seeds[post_sn_ce_mask==1]))
    RLOF_post_SN1_mask = (RLOF_SN2_mask) & (in1d(rlof_seeds, post_MT_seeds)) # systems that go SN2 and experience MT post SN1
    SN_DCO_mask = in1d(sn_seeds, dco_seeds)

    WD_mask = None
    # if include WD
    if white_dwarfs:
        WD_DCO_mask = in1d(sys_seeds, dco_seeds)
        WD_DCO_RLOF_mask = in1d(rlof_seeds, dco_seeds)
        # WD before any MT (assume mass less than 8 solar masses)
        rates_dict['ZAMS_WDBeforeMT'] = calculate_rate(sys_seeds, sys_seeds, weights, total_weight, metallicity1, unique_Z, totalMass,
                                                   condition = ((~RLOF_mask) & (stellar_merger==0) & (~SN_mask) & (~CEE_mask)),
                                                   addtnl_ce_seeds=sn_seeds[ (preSN_RLOF_mask==0) & ( ((previousStellarTypeSN>9) & (previousStellarTypeSN<13)) | ((previousStellarTypeCompanion>9) & (previousStellarTypeCompanion<13)))])
        WD_mask = rates_dict['ZAMS_WDBeforeMT']['seeds']


    # Stellar merger before any MT
    rates_dict['ZAMS_StellarMergerBeforeMT'] = calculate_rate(sys_seeds, sys_seeds, weights, total_weight, metallicity1, unique_Z, totalMass,
                                                              condition = ( (~RLOF_mask) & (stellar_merger==1) & (~SN_mask)) | ((~RLOF_mask) & (~SN_mask)),
                                                              WD_mask=WD_mask)  # if not in rlof or sn mask assume stellar merger on ZAMS


    # MT 1
    rates_dict['ZAMS_MT1'] = calculate_rate(rlof_seeds, sys_seeds, weights, total_weight, metallicity1, unique_Z, totalMass,
                                                condition = (pre_sn_mask == 1),
                                                formation_channel=fc_seeds,
                                                CEE=CE_pre_SN_seeds,
                                                WD_mask=WD_mask)
    
    if fc_seeds is not None:
        mt_other_mask = ~in1d(rlof_seeds, rates_dict['ZAMS_MT1']['seeds'])
        rates_dict['ZAMS_MT1other'] = calculate_rate(rlof_seeds, sys_seeds, weights, total_weight, metallicity1, unique_Z, totalMass,
                                                condition = ((pre_sn_mask == 1) & (mt_other_mask==1)),
                                                addtnl_ce_seeds=ce_seeds[(pre_sn_ce_mask==1) & (in1d(ce_seeds, rlof_seeds))], 
                                                CEE=CE_pre_SN_seeds,
                                                WD_mask=WD_mask)
    
    if white_dwarfs:
        # no SN WD
        SN_WD_mask = ( ((previousStellarTypeCompanion>9) & (previousStellarTypeCompanion<13)) | ((previousStellarTypeSN>9) & (previousStellarTypeCompanion<13)) )
        rlof_SN_WD_mask = in1d(rlof_seeds, sn_seeds[SN_WD_mask])

        # one component evolves into WD after MT1
        rates_dict['MT1_WD1'] = calculate_rate(rlof_seeds, sys_seeds, weights, total_weight, metallicity1, unique_Z, totalMass,
                                            condition = ((rlof_SN_WD_mask==1) | ( (pre_sn_mask==1) & (RLOF_SN_mask==0) & 
                                            ( ( ( ( (type1 > 9 ) & (type1 < 13) ) | ( (type2 > 9 ) & (type2 < 13) ) )
                                            | ( ( (type1Prev > 9 ) & (type1Prev < 13) ) | ( (type2Prev > 9 ) & (type2Prev < 13) ) ) ) ) | ((pre_sn_mask==1) & (merger_RLOF_mask==0) & (RLOF_SN_mask==0)) & (optimistic_CE_pre_sn_mask==0) )) if optimistic_CE is False else 
                                            ((rlof_SN_WD_mask==1) | ( (pre_sn_mask==1) & (RLOF_merger_pre_sn_mask==0) & (RLOF_SN_mask==0) & 
                                            ( ( ( ( (type1 > 9 ) & (type1 < 13) ) | ( (type2 > 9 ) & (type2 < 13) ) )
                                            | ( ( (type1Prev > 9 ) & (type1Prev < 13) ) | ( (type2Prev > 9 ) & (type2Prev < 13) ) ) ) ) | 
                                            ( (pre_sn_mask==1) & (RLOF_merger_pre_sn_mask==0) & (RLOF_SN_mask==0)) )),
                                            addtnl_ce_seeds=ce_seeds[(pre_sn_ce_mask==1) & (ce_stellar_merger==0) & (optimistic_CE_pre_sn_CE_mask==0) & ((ce_type1<10) & (ce_type1<10)) & ( (ce_type1_final>9) | (ce_type2_final>9))] if optimistic_CE is False else
                                               ce_seeds[(pre_sn_ce_mask==1) & (ce_stellar_merger==0) & ((ce_type1<10) & (ce_type1<10)) & ( (ce_type1_final>9) | (ce_type2_final>9))],
                                            rel_rate = rates_dict['ZAMS_MT1'],
                                            formation_channel=fc_seeds,
                                            CEE=CE_pre_SN_seeds)
        WD_mask = np.concatenate((rates_dict['ZAMS_WDBeforeMT']['seeds'], rates_dict['MT1_WD1']['seeds']))

    sm_zams = in1d(ce_seeds, rates_dict['ZAMS_StellarMergerBeforeMT']['seeds'])
    rates_dict['MT1_StellarMerger1'] = calculate_rate(rlof_seeds, sys_seeds, weights, total_weight, metallicity1, unique_Z, totalMass,
                                                          condition = ( (pre_sn_mask==1) & ( (merger_RLOF_mask==1) | (RLOF_merger_pre_sn_mask==1) ) & (RLOF_SN_mask==0) ) | (optimistic_CE_pre_sn_mask==1) if optimistic_CE is False else
                                                                    ( (pre_sn_mask==1) & ( (merger_RLOF_mask==1) | (RLOF_merger_pre_sn_mask==1) ) & (RLOF_SN_mask==0) ),
                                                          addtnl_ce_seeds=ce_seeds[ (sm_zams==0) & (pre_sn_ce_mask==1) & ((ce_stellar_merger==1) | (optimistic_CE_pre_sn_CE_mask==1))] if optimistic_CE is False else
                                                          ce_seeds[( (sm_zams==0) & (pre_sn_ce_mask==1) & (ce_stellar_merger==1))], 
                                                          rel_rate = rates_dict['ZAMS_MT1'],
                                                          formation_channel=fc_seeds,
                                                          CEE=CE_pre_SN_seeds,
                                                          WD_mask=WD_mask)
    
    # SN 1
    rates_dict['ZAMS_SN1'] = calculate_rate(sn_seeds, sys_seeds, weights, total_weight, metallicity1, unique_Z, totalMass,
                                            condition = (preSN_RLOF_mask==0),
                                            WD_mask=WD_mask)
    
    mergers = in1d(rlof_seeds, rates_dict['MT1_StellarMerger1']['seeds'])
    mergers_ce = in1d(ce_seeds, rates_dict['MT1_StellarMerger1']['seeds'])

    rates_dict['MT1_SN1'] = calculate_rate(rlof_seeds, sys_seeds, weights, total_weight, metallicity1, unique_Z, totalMass,
                                               condition = (pre_sn_mask==1) & (RLOF_SN_mask==1) & (optimistic_CE_pre_sn_mask==0) & (mergers==0) if optimistic_CE is False else 
                                                            (pre_sn_mask==1) & (RLOF_SN_mask==1) & (mergers==0),
                                               addtnl_ce_seeds=ce_seeds[(pre_sn_ce_mask==1) & (ce_stellar_merger==0) & (optimistic_CE_pre_sn_CE_mask==0) & (mergers_ce==0)] if optimistic_CE is False else
                                               ce_seeds[(pre_sn_ce_mask==1) & (ce_stellar_merger==0) & (mergers_ce==0)], 
                                               rel_rate = rates_dict['ZAMS_MT1'], 
                                               formation_channel=fc_seeds,
                                               CEE=CE_pre_SN_seeds,
                                               WD_mask=WD_mask)
    
    # SN 1 survival
    rates_dict['SN1_Disbound1'] = calculate_rate(sn_seeds, sys_seeds, weights, total_weight, metallicity1, unique_Z, totalMass,
                                                 condition = (survived==0) & (previousStellarTypeSN < 13) & (previousStellarTypeCompanion < 13) & (optimistic_CE_pre_sn_SN_mask==0) if optimistic_CE is False else
                                                            (survived==0) & (previousStellarTypeSN < 13) & (previousStellarTypeCompanion < 13), 
                                                 rel_rate=[rates_dict['MT1_SN1'],rates_dict['ZAMS_SN1']] if formation_channel is None else rates_dict['MT1_SN1'] if rates_dict['MT1_SN1']['rel_rate']!=0 else rates_dict['ZAMS_SN1'],
                                                 formation_channel=fc_seeds,
                                                 WD_mask=WD_mask)
    
    rates_dict['SN1_Survive1'] = calculate_rate(sn_seeds, sys_seeds, weights, total_weight, metallicity1, unique_Z, totalMass,
                                                 condition = (survived==1) & (previousStellarTypeSN < 13) & (previousStellarTypeCompanion < 13) & (optimistic_CE_pre_sn_SN_mask==0) if optimistic_CE is False else
                                                            (survived==1) & (previousStellarTypeSN < 13) & (previousStellarTypeCompanion < 13), 
                                                 rel_rate=[rates_dict['MT1_SN1'],rates_dict['ZAMS_SN1']] if formation_channel is None else rates_dict['MT1_SN1'] if rates_dict['MT1_SN1']['rel_rate']!=0 else rates_dict['ZAMS_SN1'],
                                                 formation_channel=fc_seeds,
                                                 WD_mask=WD_mask)

    if white_dwarfs:
        # one component evolves into WD after SN1
        rates_dict['SN1_WD2'] = calculate_rate(sn_seeds, sys_seeds, weights, total_weight, metallicity1, unique_Z, totalMass,
                                               condition = (SN_SN2_mask==0) & (~in1d(sn_seeds, rlof_seeds[RLOF_merger_pre_sn_mask==0])) & (survived==1)
                                                            & (~in1d(sn_seeds, rlof_seeds[post_sn_mask==1])) & (~in1d(sn_seeds, ce_seeds[post_sn_ce_mask==1])) & (~in1d(sn_seeds, rlof_seeds[optimistic_CE_pre_sn_mask==1])) if optimistic_CE is False else
                                                            (SN_SN2_mask==0) & (~in1d(sn_seeds, rlof_seeds[RLOF_merger_pre_sn_mask==0])) & (survived==1)
                                                            & (~in1d(sn_seeds, rlof_seeds[post_sn_mask==1])) & (~in1d(sn_seeds, ce_seeds[post_sn_ce_mask==1])),
                                                rel_rate = rates_dict['SN1_Survive1'],
                                                formation_channel=fc_seeds,
                                                CEE=CE_post_SN_seeds)
        WD_mask = np.concatenate((rates_dict['ZAMS_WDBeforeMT']['seeds'], rates_dict['MT1_WD1']['seeds'], rates_dict['SN1_WD2']['seeds']))


    if formation_channel_2 is not None:
        fc_seeds_2 = sys_seeds[formation_channel_2] if sys_mask is None else sys_seeds[formation_channel_2[sys_mask]]
    elif formation_channel_2 is None and formation_channel is not None:
        fc_seeds_2 = fc_seeds
    else:
        fc_seeds_2 = None


    rates_dict['SN1_MT2'] = calculate_rate(rlof_seeds, sys_seeds, weights, total_weight, metallicity1, unique_Z, totalMass,
                                                condition = ((post_sn_mask==1) & (optimistic_CE_pre_sn_mask==0)) if optimistic_CE is False else
                                                            (post_sn_mask==1),
                                                addtnl_ce_seeds = ce_seeds[(post_sn_ce_mask==1) & (optimistic_CE_pre_sn_CE_mask==0)] if optimistic_CE is False else
                                                ce_seeds[(post_sn_ce_mask==1)],
                                                rel_rate=rates_dict['SN1_Survive1'],
                                                formation_channel=fc_seeds_2,
                                                CEE=CE_post_SN_seeds,
                                                WD_mask=WD_mask)
    
    if white_dwarfs:
        # one component evolves into WD after MT2
        rates_dict['MT2_WD2'] = calculate_rate(rlof_seeds, sys_seeds, weights, total_weight, metallicity1, unique_Z, totalMass,
                                            condition = ( ( (post_sn_mask==1) &
                                            ( ( ( (type1 > 9 ) & (type1 < 13) ) | ( (type2 > 9 ) & (type2 < 13) ) )
                                            | ( ( (type1Prev > 9 ) & (type1Prev < 13) ) | ( (type2Prev > 9 ) & (type2Prev < 13) ) ) ) )
                                            & (optimistic_CE_post_sn_mask==0) ) if optimistic_CE is False else
                                            ( (post_sn_mask==1) &
                                            ( ( ( (type1 > 9 ) & (type1 < 13) ) | ( (type2 > 9 ) & (type2 < 13) ) )
                                            | ( ( (type1Prev > 9 ) & (type1Prev < 13) ) | ( (type2Prev > 9 ) & (type2Prev < 13) ) ) ) ),
                                            rel_rate = rates_dict['SN1_MT2'],
                                            addtnl_ce_seeds= ce_seeds[(post_sn_ce_mask==1) & ((optimistic_CE_pre_sn_CE_mask==0) & (optimistic_CE_post_sn_CE_mask==0)) & ((ce_type1<10) | (ce_type1<10)) & ( (ce_type1_final>9) | (ce_type2_final>9))] if optimistic_CE is False else
                                                ce_seeds[(post_sn_ce_mask==1) & ((ce_type1<10) | (ce_type1<10)) & ( ((ce_type1_final>9) & (ce_type1_final<13)) | ((ce_type2_final>9) & (ce_type2_final<13)) )],
                                            formation_channel=fc_seeds_2,
                                            CEE=CE_post_SN_seeds)
        WD_mask = np.concatenate((rates_dict['ZAMS_WDBeforeMT']['seeds'], rates_dict['MT1_WD1']['seeds'], rates_dict['SN1_WD2']['seeds'], rates_dict['MT2_WD2']['seeds']))

    if formation_channel_2 is not None:
        mt_other_mask = ~in1d(rlof_seeds, rates_dict['SN1_MT2']['seeds'])
        mt_other_ce_mask = ~in1d(ce_seeds, rates_dict['SN1_MT2']['seeds'])
        rates_dict['SN1_MT2other'] = calculate_rate(rlof_seeds, sys_seeds, weights, total_weight, metallicity1, unique_Z, totalMass,
                                            condition = (post_sn_mask==1) & (mt_other_mask==1) & (optimistic_CE_pre_sn_mask==0) & (optimistic_CE_pre_sn_mask==0) if optimistic_CE is False else 
                                                        (post_sn_mask==1) & (mt_other_mask==1),
                                            addtnl_ce_seeds = ce_seeds[(post_sn_ce_mask==1) & (mt_other_ce_mask==1) & (optimistic_CE_pre_sn_CE_mask==0)] if optimistic_CE is False else 
                                            ce_seeds[(post_sn_ce_mask==1) & (mt_other_ce_mask==1)],
                                            rel_rate=[rates_dict['MT1_SN1'],rates_dict['ZAMS_SN1']] if formation_channel is None else rates_dict['SN1_Survive1'],
                                            CEE=CE_post_SN_seeds,
                                            formation_channel=fc_seeds if fc_seeds is not None else None,
                                            WD_mask=WD_mask)

    rates_dict['MT2_StellarMerger2'] = calculate_rate(rlof_seeds, sys_seeds, weights, total_weight, metallicity1, unique_Z, totalMass,
                                                            condition = (( ( (post_sn_mask==1) & (merger_RLOF_mask==1) ) | (RLOF_merger_post_sn_mask==1)) | ((optimistic_CE_post_sn_mask==1) & (optimistic_CE_pre_sn_mask==0))) & (RLOF_SN_2_mask==0) if optimistic_CE is False else 
                                                                        ( ( (post_sn_mask==1) & (merger_RLOF_mask==1) ) | (RLOF_merger_post_sn_mask==1)) & (RLOF_SN_2_mask==0),
                                                            addtnl_ce_seeds = ce_seeds[((post_sn_ce_mask==1) & (ce_stellar_merger==1) & (CE_SN_2_mask==0)) | ((optimistic_CE_post_sn_CE_mask==1) & (optimistic_CE_pre_sn_CE_mask==0))] if optimistic_CE is False else 
                                                            ce_seeds[(post_sn_ce_mask==1) & (ce_stellar_merger==1) & (CE_SN_2_mask==0)],
                                                            rel_rate=rates_dict['SN1_MT2'],
                                                            formation_channel=fc_seeds_2,
                                                            CEE=CE_post_SN_seeds,
                                                            WD_mask=WD_mask)
    
    rlof_specific_seeds = np.concatenate((rates_dict['SN1_MT2']['seeds'], rates_dict['SN1_MT2other']['seeds'])) if formation_channel_2 is not None else rates_dict['SN1_MT2']['seeds']
    RLOF_postSN1_mask = in1d(rlof_seeds, rlof_specific_seeds)  # mask seeds that experience RLOF post SN1
    RLOF_SN1_SN2_mask = (RLOF_SN2_mask==1) & (RLOF_postSN1_mask==0) # mask of seeds that experience RLOF but not post SN1
    SN_SN1_SN2_mask = in1d(sn_seeds, rlof_seeds[RLOF_SN1_SN2_mask])
    SN_noRLOF_mask = ~in1d(sn_seeds, rlof_seeds)

    mergers = in1d(rlof_seeds, rates_dict['MT2_StellarMerger2']['seeds'])
    mergers_ce = in1d(ce_seeds, rates_dict['MT2_StellarMerger2']['seeds'])

    # SN1 SN2
    rates_dict['SN1_SN2'] = calculate_rate(sn_seeds, sys_seeds, weights, total_weight, metallicity1, unique_Z, totalMass,
                                            condition = (((SN_SN2_mask==1)& (SN_noRLOF_mask==1)) | (SN_SN1_SN2_mask==1)) & (optimistic_CE_pre_sn_SN_mask==0) if optimistic_CE is False else
                                                        (((SN_SN2_mask==1)& (SN_noRLOF_mask==1)) | (SN_SN1_SN2_mask==1)),
                                            rel_rate=rates_dict['SN1_Survive1'],
                                            formation_channel=fc_seeds)
    
    rates_dict['MT2_SN2'] = calculate_rate(rlof_seeds, sys_seeds, weights, total_weight, metallicity1, unique_Z, totalMass,
                                                condition = ((post_sn_mask ==1) | (RLOF_post_SN1_mask==1)) & (optimistic_CE_post_sn_mask==0) & (optimistic_CE_pre_sn_mask==0) & (RLOF_SN_2_mask==1) & (mergers==0) if optimistic_CE is False else 
                                                            ((post_sn_mask ==1) | (RLOF_post_SN1_mask==1)) & (RLOF_SN_2_mask==1) & (mergers==0),
                                                addtnl_ce_seeds = ce_seeds[((post_sn_ce_mask==1) & (ce_stellar_merger==0) & (CE_SN_2_mask==1)) & ((optimistic_CE_pre_sn_CE_mask==0) & (optimistic_CE_post_sn_CE_mask==0) & (CE_SN_2_mask==1)) & (mergers_ce==0)] if optimistic_CE is False else  
                                                ce_seeds[(post_sn_ce_mask==1) & (ce_stellar_merger==0) & (CE_SN_2_mask==1) & (mergers_ce==0)],
                                                rel_rate=rates_dict['SN1_MT2'],
                                                formation_channel=fc_seeds_2,
                                                CEE=CE_post_SN_seeds,
                                                WD_mask=WD_mask)
    
    rates_dict['SN2_Disbound2'] = calculate_rate(sn_seeds, sys_seeds, weights, total_weight, metallicity1, unique_Z, totalMass,
                                                 condition= (survived==0) & ((previousStellarTypeSN>= 13) | (previousStellarTypeCompanion >= 13)) & ((optimistic_CE_post_sn_SN_mask==0) & (optimistic_CE_pre_sn_SN_mask==0)) if optimistic_CE is False else
                                                            (survived==0) & ((previousStellarTypeSN>= 13) | (previousStellarTypeCompanion >= 13)),
                                                 rel_rate=[rates_dict['MT2_SN2'], rates_dict['SN1_SN2']] if formation_channel_2 is None else rates_dict['MT2_SN2'] if rates_dict['MT2_SN2']['rel_rate']!=0 else rates_dict['SN1_SN2'],
                                                 formation_channel=fc_seeds_2,
                                                 WD_mask=WD_mask)
    
    rates_dict['SN2_Survive2'] = calculate_rate(sn_seeds, sys_seeds, weights, total_weight, metallicity1, unique_Z, totalMass,
                                                 condition= (survived==1) & ((previousStellarTypeSN>= 13) | (previousStellarTypeCompanion >= 13)) & ((optimistic_CE_post_sn_SN_mask==0) & (optimistic_CE_pre_sn_SN_mask==0)) if optimistic_CE is False else
                                                            (survived==1) & ((previousStellarTypeSN>= 13) | (previousStellarTypeCompanion >= 13)),
                                                 rel_rate=[rates_dict['MT2_SN2'], rates_dict['SN1_SN2']] if formation_channel_2 is None else rates_dict['MT2_SN2'] if rates_dict['MT2_SN2']['rel_rate']!=0 else rates_dict['SN1_SN2'],
                                                 formation_channel=fc_seeds_2,
                                                 WD_mask=WD_mask)
    
    # DCO
    rates_dict['SN2_DCO'] = calculate_rate(sn_seeds, sys_seeds, weights, total_weight, metallicity1, unique_Z, totalMass,
                                            condition = (SN_DCO_mask==1) & ((optimistic_CE_post_sn_SN_mask==0) & (optimistic_CE_pre_sn_SN_mask==0)) if optimistic_CE is False else
                                                        (SN_DCO_mask==1),
                                            rel_rate=[rates_dict['MT2_SN2'], rates_dict['SN1_SN2']] if formation_channel_2 is None else rates_dict['MT2_SN2'],
                                            formation_channel=fc_seeds_2,
                                            WD_mask=WD_mask)
    
    # mergers
    rates_dict['DCO_Merges'] = calculate_rate(dco_seeds, sys_seeds, weights, total_weight, metallicity1, unique_Z, totalMass,
                                                condition = (mergesInHubbleTimeFlag==1) & ((optimistic_CE_post_sn_DCO_mask==0) & (optimistic_CE_pre_sn_DCO_mask==0)) if optimistic_CE is False else
                                                            (mergesInHubbleTimeFlag==1),
                                                rel_rate=rates_dict['SN2_DCO'],
                                                formation_channel=fc_seeds_2)
    rates_dict['DCO_NoMerger'] = calculate_rate(dco_seeds, sys_seeds, weights, total_weight, metallicity1, unique_Z, totalMass,
                                                condition = (mergesInHubbleTimeFlag==0)  & ((optimistic_CE_post_sn_DCO_mask==0) & (optimistic_CE_pre_sn_DCO_mask==0)) if optimistic_CE is False else
                                                            (mergesInHubbleTimeFlag==0),
                                                rel_rate=rates_dict['SN2_DCO'],
                                                formation_channel=fc_seeds_2,
                                                WD_mask=WD_mask)
    
    # calculate WD rates
    if white_dwarfs:
        
        wd_seeds = WD_mask
        wd_count = np.sum(weights[in1d(sys_seeds, wd_seeds)])
        wd_factor = wd_count/total_weight
        wd_rate = wd_count/totalMass
        WD_mask = wd_seeds
        if additional_WD_factor:
            wd_factor+=additional_WD_factor[0]
            wd_rate+=additional_WD_factor[1]


    rates_df = pd.DataFrame({
        f'rates' : {key: value['rate'] for key, value in rates_dict.items()},
        f'rel_rates' : {key: value['rel_rate'] for key, value in rates_dict.items()},
        f'counts' : {key: value['count'] for key, value in rates_dict.items()},
        f'cee_rates' : {key: value['cee_rate'] for key, value in rates_dict.items()},
        f'cee_rel_rates' : {key: value['cee_rel_rate'] for key, value in rates_dict.items()},
        f'rate_per_mass' : {key: value['rate_per_mass'] for key, value in rates_dict.items()},
        f'cee_rate_per_mass' : {key: value['cee_rate_per_mass'] for key, value in rates_dict.items()},
        f'wd_count' : [wd_count]*len(rates_dict.items()),
        f'wd_factor' : [wd_factor]*len(rates_dict.items()),
        f'wd_rate_per_mass' : [wd_rate]*len(rates_dict.items())},
        index = rates_dict.keys()
    )

    return rates_dict, rates_df
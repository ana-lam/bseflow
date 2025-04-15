from file_processing import multiprocess_files
from formation_channels import mask_mass_transfer_episodes, identify_formation_channels
from total_mass_evolved_per_Z import totalMassEvolvedPerZ
from utils import in1d
import pandas as pd
import numpy as np
import h5py
import os

class BSESimulation:
    """
    BSESimulation: Calculate stellar evolution intermediate stage rates 
    from COMPAS HDF5 outputs.
    """

    def __init__(self, filepath, selected_seeds=None, weights=None, CEE=False,
                 include_wds=False, include_pisn=False, optimistic_CE=False,
                 formation_channel=None, formation_channel_2=None):
        
        """
        Args:
            - filepath: path to the HDF5 simulation output
            - selected_seeds: list of systems to analyze
            - weights: optional array of unique weights to use
            - CEE: separate by CE or SMT for MT phases
            - include_wds: include WD systems
            - include_pisn: include (P)PISN systems
            - optimistic_CE: flag for optimistic CE model
            - formation_channel: optional formation channel mask (MT1)
            - formation_channel_2: optional formation channel mask (MT2)
        """
        
        self.file = filepath
        self.selected_seeds = selected_seeds
        self.weights = weights
        self.CEE = CEE
        self.include_wds = include_wds
        self.include_pisn = include_pisn
        self.optimistic_CE = optimistic_CE
        self.formation_channel = formation_channel
        self.formation_channel_2 = formation_channel_2

        self.wd_count = 0.0
        self.wd_rate = 0.0
        self.wd_rate_per_mass = 0.0
        self.wd_mask = None
        
        self.total_mass = None
        self.total_weight = None
        self.unique_Z = None
        self.sys_seeds = None
        self.fc_seeds = None
        self.fc_seeds_2 = None

        # cache all loaded data from HDF5 file
        self.all_data = {}

        self.rates_dict = {}
        self.rates_df = None

    def _load_all_data(self):
        """
        Preload all HDF5 groups/fields needed.
        Store in self.all_data[group][field]
        """

        # grouped list of all needed fields
        files_and_fields = [
            ('systems', ['stellar_merger', 'disbound', 'weight', 'Metallicity1']),
            ('commonEnvelopes', ['stellarType1', 'stellarType2', 'stellarMerger', 'finalStellarType1', 'finalStellarType2', 'optimisticCommonEnvelopeFlag']),
            ('RLOF', ['radius1', 'radius2', 'flagCEE', 'type1', 'type2', 'type1Prev', 'type2Prev']),
            ('supernovae', ['Survived', 'previousStellarTypeSN', 'previousStellarTypeCompanion', 'flagPISN', 'flagPPISN']),
            ('formationChannels', ['stellar_type_K1', 'stellar_type_K2']),
            ('doubleCompactObjects', ['mergesInHubbleTimeFlag'])
        ]

        # group data for one-time load
        grouped_fields = []
        for group, fields in files_and_fields:
            grouped_fields.append((self.file, group, fields, {}))
        
        # read data
        result = multiprocess_files(grouped_fields, selected_seeds=self.selected_seeds)

        self.all_data = {
            'systems': {},
            'commonEnvelopes': {},
            'RLOF': {},
            'supernovae': {},
            'formationChannels': {},
            'doubleCompactObjects': {}
        }

        idx = 0
        for file, group, fields, _ in grouped_fields:
            values = result[idx]
            for f, val in zip(fields, values[1:] if group == 'systems' else values[1:]):
                self.all_data[group][f] = val
            self.all_data[group]['seeds'] = values[0]
            idx += 1

    def get_fields(self, group, *fields):
        """
        Helper function to easily call data from self.all_data.

        Args:
            - group: which group in HDF5 ('systems', 'RLOF', 'supernovae', etc)
            - fields: field names to extract
        Returns:
            - Array or tuple of arrays of field data
        """
        values = tuple(self.all_data[group][field] for field in fields)
        return values[0] if len(values) == 1 else values    

    def clean_buggy_systems(self):

        """
        Clean buggy systems from dataset.
            - Systems with negative radii
            - Non-merging, non-unbound systems that never go SN but remain stellar type < 10
            - Systems with invalid MT
        """
        
        files_and_fields = [
            (self.file, 'systems', ['stellar_merger', 'disbound', 'weight'], {}),
            (self.file, 'commonEnvelopes', ['stellarType1', 'stellarType2', 'stellarMerger'], {}),
            (self.file, 'RLOF', ['radius1', 'radius2'], {}),
            (self.file, 'supernovae', ['Survived'], {}),
            (self.file, 'formationChannels', ['stellar_type_K1', 'stellar_type_K2'], {})
        ]

        (sys_seeds, stellar_merger, disbound, weight), \
        (ce_seeds, stellarType1, stellarType2, ce_stellar_merger), \
        (rlof_seeds, radius1, radius2),  \
        (sn_seeds, survived), \
        (f_seeds, stellar_type_K1, stellar_type_K2) = multiprocess_files(files_and_fields, selected_seeds=self.selected_seeds)

        # clean systems (negative radii)
        buggy_seeds = rlof_seeds[(radius1<0) | (radius2<0)]

        # clean more systems (don't stellar merger, don't disbound, and remain stellar type<10)
        pre_sn_mask, post_sn_mask, invalid_rlof_mask, pre_sn_ce_mask, post_sn_ce_mask = mask_mass_transfer_episodes(self.file, selected_seeds=self.selected_seeds)
        RLOF_mask = in1d(sys_seeds, rlof_seeds[~invalid_rlof_mask])


        non_dco_f_seeds = f_seeds[((stellar_type_K1<10) | (stellar_type_K2<10))]
        non_dco_sys_mask = in1d(sys_seeds, non_dco_f_seeds)

        unique_SN_seeds, sn_counts = np.unique(sn_seeds, return_counts=True)
        sys_SN2_mask = in1d(sys_seeds, unique_SN_seeds[sn_counts>1])
        merger_RLOF_seeds = sys_seeds[(RLOF_mask) & (stellar_merger==1)]
        merger_CE_seeds = ce_seeds[(ce_stellar_merger==1)]
        merger_seeds = np.concatenate((merger_RLOF_seeds, merger_CE_seeds))

        merger_mask = in1d(sys_seeds, merger_seeds)

        disregard = sys_seeds[(merger_mask==0) & (sys_SN2_mask==0)]
        sys_disregard = in1d(sys_seeds, disregard)

        more_buggy_seeds = sys_seeds[(stellar_merger==0) & (sys_disregard==1) & (disbound==0) & (non_dco_sys_mask==1)] # assume these become WD, so add to WD factor
        wd_rate = np.sum(weight[in1d(sys_seeds, more_buggy_seeds)])/np.sum(weight)

        _, total_mass = totalMassEvolvedPerZ(pathCOMPASh5=self.file, Mlower=5.0, Mupper=150.0)
        self.total_mass = np.sum(total_mass)
        wd_rate_per_mass = np.divide(np.sum(weight[in1d(sys_seeds, more_buggy_seeds)]), self.total_mass)


        buggy_seeds = np.concatenate((buggy_seeds, more_buggy_seeds))


        # clean more buggy systems, stellar type > 9 but no SN (weird accretion stuff)
        ce_seeds_in_SN = in1d(ce_seeds, sn_seeds)
        buggy_ce_seeds = ce_seeds[(ce_seeds_in_SN==0) & ( (stellarType1>9) | (stellarType2>9) )]
        
        # invalid RLOF seeds, weird MT & SN time stepping issue seeds
        invalid_RLOF_seeds = rlof_seeds[invalid_rlof_mask]

        buggy_seeds = np.concatenate((buggy_seeds, buggy_ce_seeds, invalid_RLOF_seeds))

        sys_mask = (~in1d(sys_seeds, buggy_seeds))

        self.selected_seeds = sys_seeds[sys_mask]

        # add WD to WD factor
        self.wd_rate = wd_rate
        self.wd_rate_per_mass = wd_rate_per_mass

    def specify_metallicity(self, Z, Z_max=None):

        """
        Filter systems by metallicity/metallicity range.

        Args:
            - Z: Lower bound or exact value
            - Z_max: optional upper bound
        Returns:
            - Array of seeds within Z range
        """

        files_and_fields = [
            (self.file, 'systems', ['Metallicity1'], {})
        ]

        (sys_seeds, metallicity1) = multiprocess_files(files_and_fields, selected_seeds=self.selected_seeds)

        if Z_max is not None:
            Z_mask = (metallicity1 >= Z) & (metallicity1 < Z_max)
        else:
            Z_mask = (metallicity1 == Z)

        return self.selected_seeds[Z_mask] if self.selected_seeds is not None else sys_seeds[Z_mask]
    
    def filter_by_property(self, group, prop, min_val=None, max_val=None):

        """
        Filter systems by a specific property range.
        
        Args:
            - group: HDF5 group for data
            - prop: property name (e.g., 'mass1', 'mass2', 'separation', etc)
            - min_val: Lower bound or exact value
            - max_val: optional upper bound
        Returns:
            - Array of seeds within specified property range
        """

        if prop == 'q_i':
            files_and_fields = [
                (self.file, 'systems', ['mass1', 'mass2'], {})
            ]
            (sys_seeds, mass1, mass2) = multiprocess_files(files_and_fields, selected_seeds=self.selected_seeds)
            prop_values = mass2 / mass1

        else:
            files_and_fields = [
                (self.file, f'{group}', [f'{prop}'], {})
            ]
            (sys_seeds, prop_values) = multiprocess_files(files_and_fields, selected_seeds=self.selected_seeds)

        if max_val is not None and min_val is not None:
            prop_mask = (prop_values >= min_val) & (prop_values < max_val)
        elif max_val is not None:
            prop_mask = prop_values < max_val
        elif min_val is not None:
            prop_mask = prop_values >= min_val
        else:
            return self.selected_seeds if self.selected_seeds is not None else sys_seeds
        
        return self.selected_seeds[prop_mask] if self.selected_seeds is not None else sys_seeds[prop_mask]
    
    def calculate_rate(self, data, systems, weights, total_weight, metallicities, unique_Z, total_mass,
                       CEE=None, condition=None, addtnl_ce_seeds=None, rel_rate=None, formation_channel=None, wd_mask=None):
        
        """
        Compute intermediate stage rates.

        Args:
            - data: seeds to process
            - systems: our system seeds (from `systems` HDF5 group)
            - weights: simulation weights to use
            - total_weight: sum of weights
            - metallicities: system metallicities
            - unique_Z: grid of unique metallicities in simulation
            - total_mass: total stellar mass evolved
            - CEE: split by CE or SMT for MT phases
            - condition: how to filter data to calculate intermediate stage
            - addtnl_ce_seeds: optional additional CE seeds to consider that are not in `RLOF` file
            - rel_rate: optional reference to calculate rates relative to
            - formation_channel: optional formation channel mask
            - wd_mask: mask to exclude WD systems

        Returns:
            Dict of seeds, counts, absolute/relative rates, per mass rates, CEE split rates, unique Z

        """

        def compute_rate(mask, total_weight, fc=None, addtnl_ce_seeds=None, rel_rate=None, wd_mask=None):

            masked_data = data[mask]

            if addtnl_ce_seeds is not None:
                masked_data = np.concatenate((masked_data, addtnl_ce_seeds))

            if fc is not None:
                masked_data = masked_data[in1d(masked_data, fc)]

            if wd_mask is not None:
                masked_data = masked_data[~in1d(masked_data, wd_mask)]

            mask = in1d(systems, masked_data)
            total_count = np.sum(weights[mask])
            rate = total_count / total_weight

            if rel_rate is not None:
                rel_rate_seeds = np.concatenate([r['seeds'] for r in rel_rate if 'seeds' in r]) if isinstance(rel_rate, list) else rel_rate['seeds']
                rel_rate = total_count / np.sum(weights[in1d(systems, rel_rate_seeds)])
            else:
                rel_rate = rate

            return systems[mask], total_count, rate, rel_rate
        
        def compute_rates_per_mass(seeds):
            seed_mask = in1d(systems, seeds)

            formation_rate = np.sum(weights[seed_mask]) / total_mass
            formation_rates = np.array([np.sum(weights[seed_mask & (metallicities == Z)]) for Z in unique_Z]) / total_mass

            return formation_rate, formation_rates

        mask = condition if condition is not None else np.ones_like(data, dtype=bool)

        ce_seeds = smt_seeds = ce_rate = smt_rate = ce_rel_rate = smt_rel_rate = np.nan
        ce_rate_per_mass = smt_rate_per_mass = ce_rates_per_mass = smt_rates_per_mass = np.nan

        if CEE is not None:
            ce_seeds, _, ce_rate, ce_rel_rate = compute_rate(
                mask & in1d(data, CEE), total_weight, fc=formation_channel,
                addtnl_ce_seeds=addtnl_ce_seeds[in1d(addtnl_ce_seeds, CEE)] if addtnl_ce_seeds is not None else None,
                rel_rate=rel_rate
            )
            ce_rate_per_mass, ce_rates_per_mass = compute_rates_per_mass(ce_seeds)

            smt_seeds, _, smt_rate, smt_rel_rate = compute_rate(
                mask & ~in1d(data, CEE), total_weight, fc=formation_channel,
                addtnl_ce_seeds=addtnl_ce_seeds[~in1d(addtnl_ce_seeds, CEE)] if addtnl_ce_seeds is not None else None,
                rel_rate=rel_rate
            )
            smt_rate_per_mass, smt_rates_per_mass = compute_rates_per_mass(smt_seeds)

        unique_seeds, total_count, rate, rel_rate = compute_rate(
            mask, total_weight, fc=formation_channel,
            addtnl_ce_seeds=addtnl_ce_seeds, rel_rate=rel_rate, wd_mask=wd_mask)
        rate_per_mass, rates_per_mass = compute_rates_per_mass(unique_seeds)

        return {
            'seeds': unique_seeds.astype(int), 
            'count': total_count, 
            'rate': rate, 
            'rel_rate': rel_rate, 
            'cee_rate': [ce_rate, smt_rate],
            'cee_rel_rate': [ce_rel_rate, smt_rel_rate],
            'ce_seeds': ce_seeds.astype(int) if not isinstance(ce_seeds, float) else ce_seeds,
            'smt_seeds': smt_seeds.astype(int) if not isinstance(smt_seeds, float) else smt_seeds,
            'rate_per_mass': rate_per_mass,
            'rates_per_mass': rates_per_mass,
            'cee_rate_per_mass': [ce_rate_per_mass, smt_rate_per_mass],
            'cee_rates_per_mass': [ce_rates_per_mass, smt_rates_per_mass],
            'unique_Z': unique_Z
        }
    
    def _prepare_data(self):

        """
        Apply initial filtering and prepare simulation data for rate calculation.
        """    

        sys_seeds, weights, metallicities = self.get_fields('systems', 'seeds', 'weight', 'Metallicity1')
        rlof_seeds, type1, type2, type1Prev, type2Prev = self.get_fields(
            'RLOF', 'seeds', 'type1', 'type2', 'type1Prev', 'type2Prev')
        f_seeds, stellar_type_K1, stellar_type_K2 = self.get_fields(
            'formationChannels', 'seeds', 'stellar_type_K1', 'stellar_type_K2')
        sn_seeds, previousStellarTypeSN, previousStellarTypeCompanion, flagPISN, flagPPISN = self.get_fields(
            'supernovae', 'seeds', 'previousStellarTypeSN', 'previousStellarTypeCompanion', 
            'flagPISN', 'flagPPISN')
        dco_seeds = self.get_fields('doubleCompactObjects', 'seeds')

        self.sys_seeds = sys_seeds

        # define weights if specified
        self.weights = self.weights if self.weights is not None else weights
        self.total_weight = np.sum(self.weights)

        # pull unique metallicities
        self.metallicities = metallicities
        self.unique_Z = np.unique(metallicities)
        
        sys_mask = None

        # always clean out massless remnants (stellar_type==15)
        mr_seeds = f_seeds[(stellar_type_K1==15) | (stellar_type_K2==15)]
        mr_seeds_rlof = rlof_seeds[(type1==15) | (type2==15)]
        mr_seeds = np.concatenate((mr_seeds, mr_seeds_rlof))
        mr_mask = in1d(sys_seeds, mr_seeds)
        self.rates_dict['MR'] = self.calculate_rate(self.sys_seeds, self.sys_seeds, self.weights, self.total_weight, 
                                                    self.metallicities, self.unique_Z, self.total_mass,
                                                    condition = mr_mask)
        sys_mask = (~mr_mask) 

        self.sys_seeds = sys_seeds

        # clean out WDs if not specified else wise
        # WD that goes SN (AIC NS)
        WD_SN_seeds = sn_seeds[((previousStellarTypeSN>9) & (previousStellarTypeSN<13)) | ((previousStellarTypeCompanion>9) & (previousStellarTypeCompanion<13))]
        RLOF_WD_seeds = rlof_seeds[ ( ( ( (type1 > 9 ) & (type1 < 13) ) | ( (type2 > 9 ) & (type2 < 13) ) )
                            | ( ( (type1Prev > 9 ) & (type1Prev < 13) ) | ( (type2Prev > 9 ) & (type2Prev < 13) ) ) )]

        # let's discard white dwarfs
        wd_k_type = f_seeds[ ( (stellar_type_K1>9) & (stellar_type_K1<13) ) | ( (stellar_type_K2>9) & (stellar_type_K2<13) ) ]
        white_dwarf_seeds = np.concatenate((RLOF_WD_seeds, WD_SN_seeds, wd_k_type))
        white_dwarf_seeds = white_dwarf_seeds[~in1d(white_dwarf_seeds, dco_seeds)]
        wd_mask = in1d(sys_seeds, white_dwarf_seeds)

        if self.include_wds is False:
            sys_mask = (sys_mask) & (~wd_mask)

            self.rates_dict['WD'] = self.calculate_rate(self.sys_seeds, self.sys_seeds, self.weights, self.total_weight, 
                                                    self.metallicities, self.unique_Z, self.total_mass,
                                                    condition = wd_mask & (~mr_mask))
            # save WD rate for Drake factor
            wd_rate = self.rates_dict['WD']['rate']
            wd_rate_per_mass = self.rates_dict['WD']['rate_per_mass']

            self.wd_rate += wd_rate
            self.wd_rate_per_mass += wd_rate_per_mass

        # clean out PISN systems is not specificed elsewise
        if self.include_pisn is False:
            PISN_seeds = np.concatenate((sn_seeds[flagPISN==1], sn_seeds[flagPPISN==1]))
            PISN_mask = in1d(sys_seeds, PISN_seeds)
            sys_mask = sys_mask & (~PISN_mask)
            self.rates_dict['PISN'] = self.calculate_rate(self.sys_seeds, self.sys_seeds, self.weights, self.total_weight, 
                                                    self.metallicities, self.unique_Z, self.total_mass,
                                                    condition = PISN_mask & (~mr_mask))
            
        if sys_mask is not None:
            self.sys_seeds, self.weights, self.metallicities = (self.sys_seeds[sys_mask], self.weights[sys_mask], self.metallicities[sys_mask])
            self.total_weight = np.sum(self.weights)

        if self.formation_channel is not None:
            self.fc_seeds = self.sys_seeds[self.formation_channel[sys_mask]] if sys_mask is not None else self.sys_seeds[self.formation_channel]
        else:
            self.fc_seeds = None

        if self.formation_channel_2 is not None:
            self.fc_seeds_2 = self.sys_seeds[self.formation_channel_2] if sys_mask is None else self.sys_seeds[self.formation_channel_2[sys_mask]]
        elif self.formation_channel_2 is None and self.formation_channel is not None:
            self.fc_seeds_2 = self.fc_seeds
        else:
            self.fc_seeds_2 = None


    def calculate_ZAMS(self):

        sys_seeds, stellar_merger = self.get_fields('systems', 'seeds', 'stellar_merger')
        rlof_seeds, type1, type2, type1Prev, type2Prev = self.get_fields('RLOF', 'seeds', 'type1', 'type2', 'type1Prev', 'type2Prev')
        ce_seeds = self.get_fields('commonEnvelopes', 'seeds')
        sn_seeds, previousStellarTypeSN, previousStellarTypeCompanion = self.get_fields('supernovae', 'seeds', 'previousStellarTypeSN', 'previousStellarTypeCompanion')

        # ZAMS
        self.rates_dict['ZAMS'] = self.calculate_rate(self.sys_seeds, self.sys_seeds, self.weights, self.total_weight, 
                                                      self.metallicities, self.unique_Z, self.total_mass)
        
        # useful MT masks
        pre_sn_mask, post_sn_mask, invalid_rlof_mask, pre_sn_ce_mask, post_sn_ce_mask = mask_mass_transfer_episodes(self.file, selected_seeds=sys_seeds)
        RLOF_mask = in1d(sys_seeds, rlof_seeds[~invalid_rlof_mask])
        pre_MT_seeds = np.concatenate((rlof_seeds[pre_sn_mask==1], ce_seeds[pre_sn_ce_mask==1]))
        CEE_mask = in1d(sys_seeds, ce_seeds)
        
        # useful SN masks
        SN_mask = in1d(sys_seeds, sn_seeds)
        preSN_RLOF_mask  = in1d(sn_seeds, pre_MT_seeds)

        if self.include_wds:

            self.rates_dict['ZAMS_WDBeforeMT'] = self.calculate_rate(self.sys_seeds, self.sys_seeds, self.weights, self.total_weight, 
                                                                     self.metallicities, self.unique_Z, self.total_mass,
                                                                     condition = ((~RLOF_mask) & (stellar_merger==0) & (~SN_mask) & (~CEE_mask)),
                                                                     addtnl_ce_seeds = sn_seeds[ (preSN_RLOF_mask==0) & ( ((previousStellarTypeSN>9) & 
                                                                                        (previousStellarTypeSN<13)) | ((previousStellarTypeCompanion>9) & 
                                                                                        (previousStellarTypeCompanion<13)))])
            self.wd_mask = self.rates_dict['ZAMS_WDBeforeMT']['seeds']


        # ZAMS to stellar merger at birth
        self.rates_dict['ZAMS_StellarMergerBeforeMT'] = self.calculate_rate(sys_seeds, self.sys_seeds, self.weights, self.total_weight, 
                                                                            self.metallicities, self.unique_Z, self.total_mass,
                                                                            condition = ( (~RLOF_mask) & (stellar_merger==1) & (~SN_mask)) | ((~RLOF_mask) & (~SN_mask)),
                                                                            wd_mask=self.wd_mask)

    def calculate_MT1(self):

        sys_seeds, stellar_merger = self.get_fields('systems', 'seeds', 'stellar_merger')

        rlof_seeds, flagCEE, type1, type2, type1Prev, type2Prev = self.get_fields(
            'RLOF', 'seeds', 'flagCEE', 'type1', 'type2', 'type1Prev', 'type2Prev')
        ce_seeds, ce_stellar_merger, ce_type1, ce_type2, ce_type1_final, ce_type2_final, optimisticCEflag = self.get_fields(
            'commonEnvelopes', 'seeds', 'stellarMerger', 'stellarType1', 'stellarType2',
            'finalStellarType1', 'finalStellarType2', 'optimisticCommonEnvelopeFlag')
        sn_seeds, previousStellarTypeSN, previousStellarTypeCompanion = self.get_fields(
            'supernovae', 'seeds', 'previousStellarTypeSN', 'previousStellarTypeCompanion')

        # MT masks
        pre_sn_mask, post_sn_mask, invalid_rlof_mask, pre_sn_ce_mask, post_sn_ce_mask = mask_mass_transfer_episodes(self.file, selected_seeds=self.selected_seeds)
        RLOF_mask = in1d(sys_seeds, rlof_seeds[~invalid_rlof_mask])

        # CE masks
        merger_pre_sn = ce_seeds[(ce_stellar_merger==1) & ( (ce_type1<13) | (ce_type2<13) )]
        merger_post_sn = ce_seeds[(ce_stellar_merger==1) & ( ((ce_type1>12) & (ce_type1<15)) | ((ce_type2>12) & (ce_type2<15)) )]
        RLOF_merger_pre_sn_mask = in1d(rlof_seeds, merger_pre_sn)
        RLOF_merger_post_sn_mask = in1d(rlof_seeds, merger_post_sn)

        # optimistic CE masks
        optimistic_CE_pre_sn_CE_mask = (optimisticCEflag==1) & (ce_type1 < 13) & (ce_type2 < 13)
        optimistic_CE_pre_sn_seeds = ce_seeds[optimistic_CE_pre_sn_CE_mask]
        optimistic_CE_pre_sn_mask = in1d(rlof_seeds, optimistic_CE_pre_sn_seeds)

        # SN masks
        RLOF_SN_mask = in1d(rlof_seeds, sn_seeds)

        # merger MT mask
        merger_RLOF_seeds = sys_seeds[(RLOF_mask) & (stellar_merger==1)]
        merger_RLOF_mask = in1d(rlof_seeds, merger_RLOF_seeds)

        if self.CEE:
            CE_pre_SN_seeds = rlof_seeds[(flagCEE==1) & (pre_sn_mask==1)]
            CE_pre_SN_seeds = np.concatenate((ce_seeds[pre_sn_ce_mask==1], CE_pre_SN_seeds))
            CE_post_SN_seeds = rlof_seeds[(flagCEE==1) & (post_sn_mask==1)]
            CE_post_SN_seeds = np.concatenate((ce_seeds[post_sn_ce_mask==1], CE_post_SN_seeds))
        else:
            CE_pre_SN_seeds = CE_pre_SN_mask = CE_post_SN_seeds = CE_post_SN_mask = None


        self.rates_dict['ZAMS_MT1'] = self.calculate_rate(rlof_seeds, self.sys_seeds, self.weights, self.total_weight,
                                                          self.metallicities, self.unique_Z, self.total_mass,
                                                          condition = (pre_sn_mask==1),
                                                          formation_channel= self.fc_seeds,
                                                          CEE=CE_pre_SN_seeds,
                                                          wd_mask=self.wd_mask)
        
        if self.fc_seeds is not None:
            mt_other_mask = ~in1d(rlof_seeds, self.rates_dict['ZAMS_MT1']['seeds'])
            mt_other_ce_mask = ~in1d(ce_seeds, self.rates_dict['ZAMS_MT1']['seeds'])
            self.rates_dict['ZAMS_MT1other'] = self.calculate_rate(rlof_seeds, self.sys_seeds, self.weights, self.total_weight, 
                                                                   self.metallicities, self.unique_Z, self.total_mass,
                                                                   condition = ((pre_sn_mask == 1) & (mt_other_mask==1)),
                                                                   addtnl_ce_seeds=ce_seeds[(pre_sn_ce_mask==1) & (mt_other_ce_mask==1) & (in1d(ce_seeds, rlof_seeds))], 
                                                                   CEE=CE_pre_SN_seeds,
                                                                   wd_mask=self.wd_mask)
        
        if self.include_wds:
            # no SN WD
            SN_WD_mask = ( ((previousStellarTypeCompanion>9) & (previousStellarTypeCompanion<13)) | ((previousStellarTypeSN>9) & (previousStellarTypeCompanion<13)) )
            rlof_SN_WD_mask = in1d(rlof_seeds, sn_seeds[SN_WD_mask])

            # one component evolves into WD after MT1
            self.rates_dict['MT1_WD1'] = self.calculate_rate(rlof_seeds, self.sys_seeds, self.weights, self.total_weight,
                                                            self.metallicities, self.unique_Z, self.total_mass,
                                                            condition = ((rlof_SN_WD_mask==1) | ( (pre_sn_mask==1) & (RLOF_SN_mask==0) & 
                                                            ( ( ( ( (type1 > 9 ) & (type1 < 13) ) | ( (type2 > 9 ) & (type2 < 13) ) )
                                                            | ( ( (type1Prev > 9 ) & (type1Prev < 13) ) | ( (type2Prev > 9 ) & (type2Prev < 13) ) ) ) ) | ((pre_sn_mask==1) & (merger_RLOF_mask==0) & (RLOF_SN_mask==0)) & (optimistic_CE_pre_sn_mask==0) )) if self.optimistic_CE is False else 
                                                            ((rlof_SN_WD_mask==1) | ( (pre_sn_mask==1) & (RLOF_merger_pre_sn_mask==0) & (RLOF_SN_mask==0) & 
                                                            ( ( ( ( (type1 > 9 ) & (type1 < 13) ) | ( (type2 > 9 ) & (type2 < 13) ) )
                                                            | ( ( (type1Prev > 9 ) & (type1Prev < 13) ) | ( (type2Prev > 9 ) & (type2Prev < 13) ) ) ) ) | 
                                                            ( (pre_sn_mask==1) & (RLOF_merger_pre_sn_mask==0) & (RLOF_SN_mask==0)) )),
                                                            addtnl_ce_seeds = ce_seeds[(pre_sn_ce_mask==1) & (ce_stellar_merger==0) & (optimistic_CE_pre_sn_CE_mask==0) & ((ce_type1<10) & (ce_type1<10)) & ( (ce_type1_final>9) | (ce_type2_final>9))] if self.optimistic_CE is False else
                                                            ce_seeds[(pre_sn_ce_mask==1) & (ce_stellar_merger==0) & ((ce_type1<10) & (ce_type1<10)) & ( (ce_type1_final>9) | (ce_type2_final>9))],
                                                            rel_rate = self.rates_dict['ZAMS_MT1'],
                                                            formation_channel = self.fc_seeds,
                                                            CEE = CE_pre_SN_seeds)
            self.wd_mask = np.concatenate((self.rates_dict['ZAMS_WDBeforeMT']['seeds'], self.rates_dict['MT1_WD1']['seeds']))
            
        sm_zams = in1d(ce_seeds, self.rates_dict['ZAMS_StellarMergerBeforeMT']['seeds'])
        self.rates_dict['MT1_StellarMerger1'] = self.calculate_rate(rlof_seeds, self.sys_seeds, self.weights, self.total_weight,
                                                                    self.metallicities, self.unique_Z, self.total_mass,
                                                                    condition = ( (pre_sn_mask==1) & ( (merger_RLOF_mask==1) | (RLOF_merger_pre_sn_mask==1) ) & (RLOF_SN_mask==0) ) | (optimistic_CE_pre_sn_mask==1) if self.optimistic_CE is False else
                                                                    ( (pre_sn_mask==1) & ( (merger_RLOF_mask==1) | (RLOF_merger_pre_sn_mask==1) ) & (RLOF_SN_mask==0) ),
                                                                    addtnl_ce_seeds=ce_seeds[ (sm_zams==0) & (pre_sn_ce_mask==1) & ((ce_stellar_merger==1) | (optimistic_CE_pre_sn_CE_mask==1))] if self.optimistic_CE is False else
                                                                    ce_seeds[( (sm_zams==0) & (pre_sn_ce_mask==1) & (ce_stellar_merger==1))], 
                                                                    rel_rate = self.rates_dict['ZAMS_MT1'],
                                                                    formation_channel=self.fc_seeds,
                                                                    CEE=CE_pre_SN_seeds,
                                                                    wd_mask=self.wd_mask)

    def calculate_SN1(self):

        rlof_seeds, flagCEE = self.get_fields('RLOF', 'seeds', 'flagCEE')
        ce_seeds, ce_stellar_merger, ce_type1, ce_type2, optimisticCEflag = self.get_fields(
            'commonEnvelopes', 'seeds', 'stellarMerger', 'stellarType1', 'stellarType2',
            'optimisticCommonEnvelopeFlag')
        sn_seeds, survived, previousStellarTypeSN, previousStellarTypeCompanion = self.get_fields(
            'supernovae', 'seeds', 'Survived', 'previousStellarTypeSN', 'previousStellarTypeCompanion')


        # MT masks
        pre_sn_mask, post_sn_mask, invalid_rlof_mask, pre_sn_ce_mask, post_sn_ce_mask = mask_mass_transfer_episodes(self.file, selected_seeds=self.selected_seeds)
        pre_MT_seeds = np.concatenate((rlof_seeds[pre_sn_mask==1], ce_seeds[pre_sn_ce_mask==1]))
        preSN_RLOF_mask  = in1d(sn_seeds, pre_MT_seeds)

        # CE masks
        merger_pre_sn = ce_seeds[(ce_stellar_merger==1) & ( (ce_type1<13) | (ce_type2<13) )]
        RLOF_merger_pre_sn_mask = in1d(rlof_seeds, merger_pre_sn)

        # optimistic CE masks
        optimistic_CE_pre_sn_CE_mask = (optimisticCEflag==1) & (ce_type1 < 13) & (ce_type2 < 13)
        optimistic_CE_pre_sn_seeds = ce_seeds[optimistic_CE_pre_sn_CE_mask]
        optimistic_CE_pre_sn_mask = in1d(rlof_seeds, optimistic_CE_pre_sn_seeds)
        optimistic_CE_pre_sn_SN_mask = in1d(sn_seeds, optimistic_CE_pre_sn_seeds)

        # SN masks
        RLOF_SN_mask = in1d(rlof_seeds, sn_seeds)
        unique_SN_seeds, sn_counts = np.unique(sn_seeds, return_counts=True) # systems that go SN
        SN_SN2_mask = in1d(sn_seeds, unique_SN_seeds[sn_counts>1])

        if self.CEE:
            CE_pre_SN_seeds = rlof_seeds[(flagCEE==1) & (pre_sn_mask==1)]
            CE_pre_SN_seeds = np.concatenate((ce_seeds[pre_sn_ce_mask==1], CE_pre_SN_seeds))
            CE_post_SN_seeds = rlof_seeds[(flagCEE==1) & (post_sn_mask==1)]
            CE_post_SN_seeds = np.concatenate((ce_seeds[post_sn_ce_mask==1], CE_post_SN_seeds))
        else:
            CE_pre_SN_seeds = CE_pre_SN_mask = CE_post_SN_seeds = CE_post_SN_mask = None

        self.rates_dict['ZAMS_SN1'] = self.calculate_rate(sn_seeds,  self.sys_seeds, self.weights, self.total_weight,
                                                          self.metallicities, self.unique_Z, self.total_mass,
                                                          condition = (preSN_RLOF_mask==0),
                                                          wd_mask=self.wd_mask)
        
        mergers = in1d(rlof_seeds, self.rates_dict['MT1_StellarMerger1']['seeds'])
        mergers_ce = in1d(ce_seeds, self.rates_dict['MT1_StellarMerger1']['seeds'])

        self.rates_dict['MT1_SN1'] = self.calculate_rate(rlof_seeds, self.sys_seeds, self.weights, self.total_weight,
                                                         self.metallicities, self.unique_Z, self.total_mass,
                                                         condition = (pre_sn_mask==1) & (RLOF_SN_mask==1) & (optimistic_CE_pre_sn_mask==0) & (mergers==0) if self.optimistic_CE is False else 
                                                        (pre_sn_mask==1) & (RLOF_SN_mask==1) & (mergers==0),
                                                         addtnl_ce_seeds=ce_seeds[(pre_sn_ce_mask==1) & (ce_stellar_merger==0) & (optimistic_CE_pre_sn_CE_mask==0) & (mergers_ce==0)] if self.optimistic_CE is False else
                                                         ce_seeds[(pre_sn_ce_mask==1) & (ce_stellar_merger==0) & (mergers_ce==0)], 
                                                         rel_rate = self.rates_dict['ZAMS_MT1'], 
                                                         formation_channel=self.fc_seeds,
                                                         CEE=CE_pre_SN_seeds,
                                                         wd_mask=self.wd_mask)

        self.rates_dict['SN1_Disbound1'] = self.calculate_rate(sn_seeds, self.sys_seeds, self.weights, self.total_weight,
                                                               self.metallicities, self.unique_Z, self.total_mass,
                                                               condition = (survived==0) & (previousStellarTypeSN < 13) & (previousStellarTypeCompanion < 13) & (optimistic_CE_pre_sn_SN_mask==0) if self.optimistic_CE is False else
                                                               (survived==0) & (previousStellarTypeSN < 13) & (previousStellarTypeCompanion < 13), 
                                                               rel_rate=[self.rates_dict['MT1_SN1'], self.rates_dict['ZAMS_SN1']] if self.formation_channel is None else self.rates_dict['MT1_SN1'] if self.rates_dict['MT1_SN1']['rel_rate']!=0 else self.rates_dict['ZAMS_SN1'],
                                                               formation_channel=self.fc_seeds,
                                                               wd_mask=self.wd_mask)
    
        self.rates_dict['SN1_Survive1'] = self.calculate_rate(sn_seeds, self.sys_seeds, self.weights, self.total_weight,
                                                              self.metallicities, self.unique_Z, self.total_mass,
                                                              condition = (survived==1) & (previousStellarTypeSN < 13) & (previousStellarTypeCompanion < 13) & (optimistic_CE_pre_sn_SN_mask==0) if self.optimistic_CE is False else
                                                              (survived==1) & (previousStellarTypeSN < 13) & (previousStellarTypeCompanion < 13), 
                                                              rel_rate=[self.rates_dict['MT1_SN1'], self.rates_dict['ZAMS_SN1']] if self.formation_channel is None else self.rates_dict['MT1_SN1'] if self.rates_dict['MT1_SN1']['rel_rate']!=0 else self.rates_dict['ZAMS_SN1'],
                                                              formation_channel=self.fc_seeds,
                                                              wd_mask=self.wd_mask)
        
        if self.include_wds:
            # one component evolves into WD after SN1
            self.rates_dict['SN1_WD2'] = self.calculate_rate(sn_seeds, self.sys_seeds, self.weights, self.total_weight,
                                                             self.metallicities, self.unique_Z, self.total_mass,
                                                             condition = (SN_SN2_mask==0) & (~in1d(sn_seeds, rlof_seeds[RLOF_merger_pre_sn_mask==0])) & (survived==1)
                                                             & (~in1d(sn_seeds, rlof_seeds[post_sn_mask==1])) & (~in1d(sn_seeds, ce_seeds[post_sn_ce_mask==1])) & (~in1d(sn_seeds, rlof_seeds[optimistic_CE_pre_sn_mask==1])) if self.optimistic_CE is False else
                                                             (SN_SN2_mask==0) & (~in1d(sn_seeds, rlof_seeds[RLOF_merger_pre_sn_mask==0])) & (survived==1)
                                                             & (~in1d(sn_seeds, rlof_seeds[post_sn_mask==1])) & (~in1d(sn_seeds, ce_seeds[post_sn_ce_mask==1])),
                                                             rel_rate = self.rates_dict['SN1_Survive1'],
                                                             formation_channel=self.fc_seeds,
                                                             CEE=CE_post_SN_seeds)
            self.wd_mask = np.concatenate((self.rates_dict['ZAMS_WDBeforeMT']['seeds'], self.rates_dict['MT1_WD1']['seeds'], self.rates_dict['SN1_WD2']['seeds']))

    def calculate_MT2(self):

        sys_seeds, stellar_merger = self.get_fields('systems', 'seeds', 'stellar_merger')
        rlof_seeds, flagCEE, type1, type2, type1Prev, type2Prev = self.get_fields(
            'RLOF', 'seeds', 'flagCEE', 'type1', 'type2', 'type1Prev', 'type2Prev')
        ce_seeds, ce_stellar_merger, ce_type1, ce_type2, ce_type1_final, ce_type2_final, optimisticCEflag = self.get_fields(
            'commonEnvelopes', 'seeds', 'stellarMerger', 'stellarType1', 'stellarType2',
            'finalStellarType1', 'finalStellarType2', 'optimisticCommonEnvelopeFlag')
        sn_seeds= self.get_fields('supernovae', 'seeds')

        # MT masks
        pre_sn_mask, post_sn_mask, invalid_rlof_mask, pre_sn_ce_mask, post_sn_ce_mask = mask_mass_transfer_episodes(self.file, selected_seeds=self.selected_seeds)
        RLOF_mask = in1d(sys_seeds, rlof_seeds[~invalid_rlof_mask])

        # CE masks
        merger_post_sn = ce_seeds[(ce_stellar_merger==1) & ( ((ce_type1>12) & (ce_type1<15)) | ((ce_type2>12) & (ce_type2<15)) )]
        RLOF_merger_post_sn_mask = in1d(rlof_seeds, merger_post_sn)

        # merger MT mask
        merger_RLOF_seeds = sys_seeds[(RLOF_mask) & (stellar_merger==1)]
        merger_RLOF_mask = in1d(rlof_seeds, merger_RLOF_seeds)

        # optimistic CE masks
        optimistic_CE_pre_sn_CE_mask = (optimisticCEflag==1) & (ce_type1 < 13) & (ce_type2 < 13)
        optimistic_CE_post_sn_CE_mask = (optimisticCEflag==1) & ( (ce_type1>12) | (ce_type2>12))

        optimistic_CE_pre_sn_seeds = ce_seeds[optimistic_CE_pre_sn_CE_mask]
        optimistic_CE_post_sn_seeds = ce_seeds[optimistic_CE_post_sn_CE_mask]
        
        optimistic_CE_pre_sn_mask = in1d(rlof_seeds, optimistic_CE_pre_sn_seeds)
        optimistic_CE_post_sn_mask = in1d(rlof_seeds, optimistic_CE_post_sn_seeds)

        # SN masks
        unique_SN_seeds, sn_counts = np.unique(sn_seeds, return_counts=True)
        SN_SN2_mask = in1d(sn_seeds, unique_SN_seeds[sn_counts>1])
        RLOF_SN_2_mask = in1d(rlof_seeds, sn_seeds[SN_SN2_mask])
        CE_SN_2_mask = in1d(ce_seeds, sn_seeds[SN_SN2_mask])

        if self.CEE:
            CE_pre_SN_seeds = rlof_seeds[(flagCEE==1) & (pre_sn_mask==1)]
            CE_pre_SN_seeds = np.concatenate((ce_seeds[pre_sn_ce_mask==1], CE_pre_SN_seeds))
            CE_post_SN_seeds = rlof_seeds[(flagCEE==1) & (post_sn_mask==1)]
            CE_post_SN_seeds = np.concatenate((ce_seeds[post_sn_ce_mask==1], CE_post_SN_seeds))
        else:
            CE_pre_SN_seeds = CE_pre_SN_mask = CE_post_SN_seeds = CE_post_SN_mask = None

        self.rates_dict['SN1_MT2'] = self.calculate_rate(rlof_seeds, self.sys_seeds, self.weights, self.total_weight,
                                                         self.metallicities, self.unique_Z, self.total_mass,
                                                         condition = ((post_sn_mask==1) & (optimistic_CE_pre_sn_mask==0)) if self.optimistic_CE is False else
                                                         (post_sn_mask==1),
                                                         addtnl_ce_seeds = ce_seeds[(post_sn_ce_mask==1) & (optimistic_CE_pre_sn_CE_mask==0)] if self.optimistic_CE is False else
                                                         ce_seeds[(post_sn_ce_mask==1)],
                                                         rel_rate=self.rates_dict['SN1_Survive1'],
                                                         formation_channel=self.fc_seeds_2,
                                                         CEE=CE_post_SN_seeds,
                                                         wd_mask=self.wd_mask)
        
        if self.include_wds:
            # one component evolves into WD after MT2
            self.rates_dict['MT2_WD2'] = self.calculate_rate(rlof_seeds, self.sys_seeds, self.weights, self.total_weight,
                                                             self.metallicities, self.unique_Z, self.total_mass,
                                                             condition = ( ( (post_sn_mask==1) &
                                                             ( ( ( (type1 > 9 ) & (type1 < 13) ) | ( (type2 > 9 ) & (type2 < 13) ) )
                                                             | ( ( (type1Prev > 9 ) & (type1Prev < 13) ) | ( (type2Prev > 9 ) & (type2Prev < 13) ) ) ) )
                                                             & (optimistic_CE_post_sn_mask==0) ) if self.optimistic_CE is False else
                                                             ( (post_sn_mask==1) &
                                                             ( ( ( (type1 > 9 ) & (type1 < 13) ) | ( (type2 > 9 ) & (type2 < 13) ) )
                                                             | ( ( (type1Prev > 9 ) & (type1Prev < 13) ) | ( (type2Prev > 9 ) & (type2Prev < 13) ) ) ) ),
                                                             rel_rate = self.rates_dict['SN1_MT2'],
                                                             addtnl_ce_seeds= ce_seeds[(post_sn_ce_mask==1) & ((optimistic_CE_pre_sn_CE_mask==0) & (optimistic_CE_post_sn_CE_mask==0)) & ((ce_type1<10) | (ce_type1<10)) & ( (ce_type1_final>9) | (ce_type2_final>9))] if self.optimistic_CE is False else
                                                             ce_seeds[(post_sn_ce_mask==1) & ((ce_type1<10) | (ce_type1<10)) & ( ((ce_type1_final>9) & (ce_type1_final<13)) | ((ce_type2_final>9) & (ce_type2_final<13)) )],
                                                             formation_channel=self.fc_seeds_2,
                                                             CEE=CE_post_SN_seeds)
            self.wd_mask = np.concatenate((self.rates_dict['ZAMS_WDBeforeMT']['seeds'], self.rates_dict['MT1_WD1']['seeds'], self.rates_dict['SN1_WD2']['seeds'], self.rates_dict['MT2_WD2']['seeds']))

        if self.formation_channel_2 is not None:
            mt_other_mask = ~in1d(rlof_seeds, self.rates_dict['SN1_MT2']['seeds'])
            mt_other_ce_mask = ~in1d(ce_seeds, self.rates_dict['SN1_MT2']['seeds'])
            self.rates_dict['SN1_MT2other'] = self.calculate_rate(rlof_seeds, self.sys_seeds, self.weights, self.total_weight,
                                                                 self.metallicities, self.unique_Z, self.total_mass,
                                                                 condition = (post_sn_mask==1) & (mt_other_mask==1) & (optimistic_CE_pre_sn_mask==0) & (optimistic_CE_pre_sn_mask==0) if self.optimistic_CE is False else 
                                                                 (post_sn_mask==1) & (mt_other_mask==1),
                                                                 addtnl_ce_seeds = ce_seeds[(post_sn_ce_mask==1) & (mt_other_ce_mask==1) & (optimistic_CE_pre_sn_CE_mask==0)] if self.optimistic_CE is False else 
                                                                 ce_seeds[(post_sn_ce_mask==1) & (mt_other_ce_mask==1)],
                                                                 rel_rate=[self.rates_dict['MT1_SN1'],self.rates_dict['ZAMS_SN1']] if self.formation_channel is None else self.rates_dict['SN1_Survive1'],
                                                                 CEE=CE_post_SN_seeds,
                                                                 formation_channel=self.fc_seeds if self.fc_seeds is not None else None,
                                                                 wd_mask=self.wd_mask)
            
        self.rates_dict['MT2_StellarMerger2'] = self.calculate_rate(rlof_seeds, self.sys_seeds, self.weights, self.total_weight,
                                                                    self.metallicities, self.unique_Z, self.total_mass,
                                                                    condition = (( ( (post_sn_mask==1) & (merger_RLOF_mask==1) ) | (RLOF_merger_post_sn_mask==1)) | ((optimistic_CE_post_sn_mask==1) & (optimistic_CE_pre_sn_mask==0))) & (RLOF_SN_2_mask==0) if self.optimistic_CE is False else 
                                                                    ( ( (post_sn_mask==1) & (merger_RLOF_mask==1) ) | (RLOF_merger_post_sn_mask==1)) & (RLOF_SN_2_mask==0),
                                                                    addtnl_ce_seeds = ce_seeds[((post_sn_ce_mask==1) & (ce_stellar_merger==1) & (CE_SN_2_mask==0)) | ((optimistic_CE_post_sn_CE_mask==1) & (optimistic_CE_pre_sn_CE_mask==0))] if self.optimistic_CE is False else 
                                                                    ce_seeds[(post_sn_ce_mask==1) & (ce_stellar_merger==1) & (CE_SN_2_mask==0)],
                                                                    rel_rate=self.rates_dict['SN1_MT2'],
                                                                    formation_channel=self.fc_seeds_2,
                                                                    CEE=CE_post_SN_seeds,
                                                                    wd_mask=self.wd_mask)

    def calculate_SN2(self):

        sys_seeds = self.get_fields('systems', 'seeds')
        rlof_seeds, flagCEE = self.get_fields(
            'RLOF', 'seeds', 'flagCEE')
        ce_seeds, ce_stellar_merger, ce_type1, ce_type2, optimisticCEflag = self.get_fields(
            'commonEnvelopes', 'seeds', 'stellarMerger', 'stellarType1', 'stellarType2',
            'optimisticCommonEnvelopeFlag')
        sn_seeds, survived, previousStellarTypeSN, previousStellarTypeCompanion = self.get_fields(
            'supernovae', 'seeds', 'Survived', 'previousStellarTypeSN', 'previousStellarTypeCompanion')
        dco_seeds = self.get_fields('doubleCompactObjects', 'seeds')

        # MT masks
        pre_sn_mask, post_sn_mask, invalid_rlof_mask, pre_sn_ce_mask, post_sn_ce_mask = mask_mass_transfer_episodes(self.file, selected_seeds=self.selected_seeds)
        RLOF_mask = in1d(sys_seeds, rlof_seeds[~invalid_rlof_mask])

        # optimistic CE masks
        optimistic_CE_pre_sn_CE_mask = (optimisticCEflag==1) & (ce_type1 < 13) & (ce_type2 < 13)
        optimistic_CE_post_sn_CE_mask = (optimisticCEflag==1) & ( (ce_type1>12) | (ce_type2>12))

        optimistic_CE_pre_sn_seeds = ce_seeds[optimistic_CE_pre_sn_CE_mask]
        optimistic_CE_post_sn_seeds = ce_seeds[optimistic_CE_post_sn_CE_mask]

        optimistic_CE_pre_sn_mask = in1d(rlof_seeds, optimistic_CE_pre_sn_seeds)
        optimistic_CE_post_sn_mask = in1d(rlof_seeds, optimistic_CE_post_sn_seeds)

        optimistic_CE_pre_sn_SN_mask = in1d(sn_seeds, optimistic_CE_pre_sn_seeds)
        optimistic_CE_post_sn_SN_mask = in1d(sn_seeds, optimistic_CE_post_sn_seeds)

        # SN masks
        unique_SN_seeds, sn_counts = np.unique(sn_seeds, return_counts=True)
        SN_SN2_mask = in1d(sn_seeds, unique_SN_seeds[sn_counts>1])
        RLOF_SN2_mask = in1d(rlof_seeds, unique_SN_seeds[sn_counts>1])
        RLOF_SN_2_mask = in1d(rlof_seeds, sn_seeds[SN_SN2_mask])
        CE_SN_2_mask = in1d(ce_seeds, sn_seeds[SN_SN2_mask])
        post_MT_seeds = np.concatenate((rlof_seeds[post_sn_mask==1], ce_seeds[post_sn_ce_mask==1]))
        RLOF_post_SN1_mask = (RLOF_SN2_mask) & (in1d(rlof_seeds, post_MT_seeds)) # systems that go SN2 and experience MT post SN1
        SN_DCO_mask = in1d(sn_seeds, dco_seeds)

        if self.CEE:
            CE_pre_SN_seeds = rlof_seeds[(flagCEE==1) & (pre_sn_mask==1)]
            CE_pre_SN_seeds = np.concatenate((ce_seeds[pre_sn_ce_mask==1], CE_pre_SN_seeds))
            CE_post_SN_seeds = rlof_seeds[(flagCEE==1) & (post_sn_mask==1)]
            CE_post_SN_seeds = np.concatenate((ce_seeds[post_sn_ce_mask==1], CE_post_SN_seeds))
        else:
            CE_pre_SN_seeds = CE_pre_SN_mask = CE_post_SN_seeds = CE_post_SN_mask = None

        # unique masks
        rlof_specific_seeds = np.concatenate((self.rates_dict['SN1_MT2']['seeds'], self.rates_dict['SN1_MT2other']['seeds'])) if self.formation_channel_2 is not None else self.rates_dict['SN1_MT2']['seeds']
        RLOF_postSN1_mask = in1d(rlof_seeds, rlof_specific_seeds)  # mask seeds that experience RLOF post SN1
        RLOF_SN1_SN2_mask = (RLOF_SN2_mask==1) & (RLOF_postSN1_mask==0) # mask of seeds that experience RLOF but not post SN1
        SN_SN1_SN2_mask = in1d(sn_seeds, rlof_seeds[RLOF_SN1_SN2_mask])
        SN_noRLOF_mask = ~in1d(sn_seeds, rlof_seeds)
        

        mergers = in1d(rlof_seeds, self.rates_dict['MT2_StellarMerger2']['seeds'])
        mergers_ce = in1d(ce_seeds, self.rates_dict['MT2_StellarMerger2']['seeds'])

        self.rates_dict['SN1_SN2'] = self.calculate_rate(sn_seeds, self.sys_seeds, self.weights, self.total_weight,
                                                        self.metallicities, self.unique_Z, self.total_mass,
                                                        condition = (((SN_SN2_mask==1)& (SN_noRLOF_mask==1)) | (SN_SN1_SN2_mask==1)) & (optimistic_CE_pre_sn_SN_mask==0) if self.optimistic_CE is False else
                                                        (((SN_SN2_mask==1)& (SN_noRLOF_mask==1)) | (SN_SN1_SN2_mask==1)),
                                                        rel_rate=self.rates_dict['SN1_Survive1'],
                                                        formation_channel=self.fc_seeds)
        
        self.rates_dict['MT2_SN2'] = self.calculate_rate(rlof_seeds, self.sys_seeds, self.weights, self.total_weight,
                                                        self.metallicities, self.unique_Z, self.total_mass,
                                                        condition = ((post_sn_mask ==1) | (RLOF_post_SN1_mask==1)) & (optimistic_CE_post_sn_mask==0) & (optimistic_CE_pre_sn_mask==0) & (RLOF_SN_2_mask==1) & (mergers==0) if self.optimistic_CE is False else 
                                                        ((post_sn_mask ==1) | (RLOF_post_SN1_mask==1)) & (RLOF_SN_2_mask==1) & (mergers==0),
                                                        addtnl_ce_seeds = ce_seeds[((post_sn_ce_mask==1) & (ce_stellar_merger==0) & (CE_SN_2_mask==1)) & ((optimistic_CE_pre_sn_CE_mask==0) & (optimistic_CE_post_sn_CE_mask==0) & (CE_SN_2_mask==1)) & (mergers_ce==0)] if self.optimistic_CE is False else  
                                                        ce_seeds[(post_sn_ce_mask==1) & (ce_stellar_merger==0) & (CE_SN_2_mask==1) & (mergers_ce==0)],
                                                        rel_rate=self.rates_dict['SN1_MT2'],
                                                        formation_channel=self.fc_seeds_2,
                                                        CEE=CE_post_SN_seeds,
                                                        wd_mask=self.wd_mask)
        
        self.rates_dict['SN2_Disbound2'] = self.calculate_rate(sn_seeds, self.sys_seeds, self.weights, self.total_weight,
                                                               self.metallicities, self.unique_Z, self.total_mass,
                                                               condition= (survived==0) & ((previousStellarTypeSN>= 13) | (previousStellarTypeCompanion >= 13)) & ((optimistic_CE_post_sn_SN_mask==0) & (optimistic_CE_pre_sn_SN_mask==0)) if self.optimistic_CE is False else
                                                               (survived==0) & ((previousStellarTypeSN>= 13) | (previousStellarTypeCompanion >= 13)),
                                                               rel_rate=[self.rates_dict['MT2_SN2'], self.rates_dict['SN1_SN2']] if self.formation_channel_2 is None else self.rates_dict['MT2_SN2'] if self.rates_dict['MT2_SN2']['rel_rate']!=0 else self.rates_dict['SN1_SN2'],
                                                               formation_channel=self.fc_seeds_2,
                                                               wd_mask=self.wd_mask)
    
        self.rates_dict['SN2_Survive2'] = self.calculate_rate(sn_seeds, self.sys_seeds, self.weights, self.total_weight,
                                                              self.metallicities, self.unique_Z, self.total_mass,
                                                              condition= (survived==1) & ((previousStellarTypeSN>= 13) | (previousStellarTypeCompanion >= 13)) & ((optimistic_CE_post_sn_SN_mask==0) & (optimistic_CE_pre_sn_SN_mask==0)) if self.optimistic_CE is False else
                                                              (survived==1) & ((previousStellarTypeSN>= 13) | (previousStellarTypeCompanion >= 13)),
                                                              rel_rate=[self.rates_dict['MT2_SN2'], self.rates_dict['SN1_SN2']] if self.formation_channel_2 is None else self.rates_dict['MT2_SN2'] if self.rates_dict['MT2_SN2']['rel_rate']!=0 else self.rates_dict['SN1_SN2'],
                                                              formation_channel=self.fc_seeds_2,
                                                              wd_mask=self.wd_mask)

        self.rates_dict['SN2_DCO'] = self.calculate_rate(sn_seeds, self.sys_seeds, self.weights, self.total_weight,
                                                        self.metallicities, self.unique_Z, self.total_mass,
                                                        condition = (SN_DCO_mask==1) & ((optimistic_CE_post_sn_SN_mask==0) & (optimistic_CE_pre_sn_SN_mask==0)) if self.optimistic_CE is False else
                                                                    (SN_DCO_mask==1),
                                                        rel_rate=[self.rates_dict['MT2_SN2'], self.rates_dict['SN1_SN2']] if self.formation_channel_2 is None else self.rates_dict['MT2_SN2'],
                                                        formation_channel=self.fc_seeds_2,
                                                        wd_mask=self.wd_mask)
        

    def calculate_DCO(self):

        ce_seeds, ce_type1, ce_type2, optimisticCEflag = self.get_fields(
            'commonEnvelopes', 'seeds', 'stellarType1', 'stellarType2',
            'optimisticCommonEnvelopeFlag')
        dco_seeds, mergesInHubbleTimeFlag = self.get_fields('doubleCompactObjects', 'seeds', 'mergesInHubbleTimeFlag')

        # optimistic CE flags
        optimistic_CE_pre_sn_CE_mask = (optimisticCEflag==1) & (ce_type1 < 13) & (ce_type2 < 13)
        optimistic_CE_post_sn_CE_mask = (optimisticCEflag==1) & ( (ce_type1>12) | (ce_type2>12))

        optimistic_CE_pre_sn_seeds = ce_seeds[optimistic_CE_pre_sn_CE_mask]
        optimistic_CE_post_sn_seeds = ce_seeds[optimistic_CE_post_sn_CE_mask]

        optimistic_CE_pre_sn_DCO_mask = in1d(dco_seeds, optimistic_CE_pre_sn_seeds)
        optimistic_CE_post_sn_DCO_mask = in1d(dco_seeds, optimistic_CE_post_sn_seeds) 


        self.rates_dict['DCO_Merges'] = self.calculate_rate(dco_seeds, self.sys_seeds, self.weights, self.total_weight,
                                                           self.metallicities, self.unique_Z, self.total_mass,
                                                           condition = (mergesInHubbleTimeFlag==1) & ((optimistic_CE_post_sn_DCO_mask==0) & (optimistic_CE_pre_sn_DCO_mask==0)) if self.optimistic_CE is False else
                                                           (mergesInHubbleTimeFlag==1),
                                                           rel_rate=self.rates_dict['SN2_DCO'],
                                                           formation_channel=self.fc_seeds_2,
                                                           wd_mask=self.wd_mask)
                                                           
        self.rates_dict['DCO_NoMerger'] = self.calculate_rate(dco_seeds, self.sys_seeds, self.weights, self.total_weight,
                                                            self.metallicities, self.unique_Z, self.total_mass,
                                                            condition = (mergesInHubbleTimeFlag==0)  & ((optimistic_CE_post_sn_DCO_mask==0) & (optimistic_CE_pre_sn_DCO_mask==0)) if self.optimistic_CE is False else
                                                            (mergesInHubbleTimeFlag==0),
                                                            rel_rate=self.rates_dict['SN2_DCO'],
                                                            formation_channel=self.fc_seeds_2,
                                                            wd_mask=self.wd_mask)

    def calculate_all_rates(self):

        """
        Perform all methods to yield results dictionary and dataframe.
        """
        
        # clean data
        self.clean_buggy_systems()

        if self.formation_channel is not None:
            channels, masks = identify_formation_channels(self.file, selected_seeds=self.selected_seeds)
            self.formation_channel = masks[self.formation_channel]
            if self.formation_channel_2 is not None:
                self.formation_channel_2 = masks[self.formation_channel_2]

        # cache data
        self._load_all_data()
        # prepare data for calculations
        self._prepare_data()

        # calculate rates
        self.calculate_ZAMS()
        self.calculate_MT1()
        self.calculate_SN1()
        self.calculate_MT2()
        self.calculate_SN2()
        self.calculate_DCO()

        if self.include_wds:

            wd_seeds = self.wd_mask
            self.wd_count = np.sum(self.weights[in1d(self.sys_seeds, wd_seeds)])
            wd_rate = self.wd_count/self.total_weight
            wd_rate_per_mass = self.wd_count/self.total_mass
            
            self.wd_rate +=  wd_rate
            self.wd_rate_per_mass += wd_rate_per_mass

        self._build_dataframe()
        return self.rates_dict, self.rates_df

    def _build_dataframe(self):

        """
        Build dataframe with intermediate stage rates.
        """
        
        self.rates_df = pd.DataFrame({
            f'rates' : {key: value['rate'] for key, value in self.rates_dict.items()},
            f'rel_rates' : {key: value['rel_rate'] for key, value in self.rates_dict.items()},
            f'counts' : {key: value['count'] for key, value in self.rates_dict.items()},
            f'cee_rates' : {key: value['cee_rate'] for key, value in self.rates_dict.items()},
            f'cee_rel_rates' : {key: value['cee_rel_rate'] for key, value in self.rates_dict.items()},
            f'rate_per_mass' : {key: value['rate_per_mass'] for key, value in self.rates_dict.items()},
            f'cee_rate_per_mass' : {key: value['cee_rate_per_mass'] for key, value in self.rates_dict.items()},
            f'wd_count' : [self.wd_count]*len(self.rates_dict.items()),
            f'wd_factor' : [self.wd_rate]*len(self.rates_dict.items()),
            f'wd_rate_per_mass' : [self.wd_rate_per_mass]*len(self.rates_dict.items())},
            index = self.rates_dict.keys()
            )
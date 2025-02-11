#!/usr/bin/env python3

import argparse
import os
from calculate_rates import calculate_simulation_rates, clean_buggy_systems, specify_metallicity, specify_masses
from formation_channels import identify_formation_channels
import h5py as h5
import numpy as np
from file_processing import create_h5_file


def output_results(file, save_path, CEE=False, Z=None, Z_max=None, m_min=None, m_max=None, MT1mask=None, MT2mask=None, selected_seeds=None, write_seeds=None):

    print("starting...")

    model = file.split("/")[-2] # record model from file path name

    cleaned_seeds, WD_factor, WD_rate = clean_buggy_systems(file, selected_seeds=selected_seeds)

    if Z:
        cleaned_seeds = specify_metallicity(file, Z, Z_max=Z_max, selected_seeds=cleaned_seeds)
    if m_min or m_max:
        cleaned_seeds = specify_masses(file, m_min=m_min, m_max=m_max, selected_seeds=cleaned_seeds)

    print("calculating channels...")
    if MT1mask is not None:
        channels, masks = identify_formation_channels(file, selected_seeds=cleaned_seeds)
    
    print("calculating rates...")

    if MT1mask is None:
        rates_dict_wd, rates_df_wd = calculate_simulation_rates(file, formation_channel=None, formation_channel_2=None, 
                                                                selected_seeds=cleaned_seeds, white_dwarfs=True, 
                                                                additional_WD_factor=[WD_factor, WD_rate], CEE=CEE)
        rates_dict, rates_df = calculate_simulation_rates(file, formation_channel=None, formation_channel_2=None, 
                                                          selected_seeds=cleaned_seeds, additional_WD_factor=[WD_factor, WD_rate], 
                                                          CEE=CEE)
    elif MT2mask is None:
        rates_dict, rates_df = calculate_simulation_rates(file, formation_channel=masks[MT1mask], formation_channel_2=None, 
                                                          selected_seeds=cleaned_seeds, additional_WD_factor=[WD_factor, WD_rate])
    else:
        rates_dict, rates_df = calculate_simulation_rates(file, formation_channel=masks[MT1mask], 
                                                          formation_channel_2=(masks[MT1mask] & masks[MT2mask]), 
                                                          selected_seeds=cleaned_seeds, additional_WD_factor=[WD_factor, WD_rate])

    print("writing csv...")
    rates_df.to_csv(f'../data/rates_{model}_{save_path}.csv')
    if write_seeds:
        create_h5_file(f'../data/fc_stage_seeds/{write_seeds}_{model}.h5', rates_dict)

    if 'rates_df_wd' in locals() and not rates_df_wd.empty and rates_df_wd.values.any():
        rates_df_wd.to_csv(f'../data/rates_{model}_{save_path}_WD.csv')

    if model == 'fiducial' or model == 'unstableCaseBB':
        print("running calculation with optimistic CE...")
        print("cleaning out buggy systems...")
        cleaned_seeds, WD_factor, WD_rate = clean_buggy_systems(file, selected_seeds=None)
        print("calculating channels...")
        channels, masks = identify_formation_channels(file, selected_seeds=cleaned_seeds)
        print("calculating rates...")

        if MT1mask is None:
            rates_dict_wd, rates_df_wd = calculate_simulation_rates(file, formation_channel=None, formation_channel_2=None, 
                                                                    optimistic_CE=True, selected_seeds=cleaned_seeds, white_dwarfs=True, 
                                                                    additional_WD_factor=[WD_factor, WD_rate], CEE=CEE)
            rates_dict, rates_df = calculate_simulation_rates(file, formation_channel=None, formation_channel_2=None, 
                                                              optimistic_CE=True, selected_seeds=cleaned_seeds, 
                                                              additional_WD_factor=[WD_factor, WD_rate], CEE=CEE)
        elif MT2mask is None:
            rates_dict, rates_df = calculate_simulation_rates(file, formation_channel=masks[MT1mask], formation_channel_2=None, 
                                                              optimistic_CE=True, selected_seeds=cleaned_seeds, 
                                                              additional_WD_factor=[WD_factor, WD_rate])
        else:
            rates_dict, rates_df = calculate_simulation_rates(file, formation_channel=masks[MT1mask], 
                                                              formation_channel_2=(masks[MT1mask] & masks[MT2mask]), 
                                                              optimistic_CE=True, selected_seeds=cleaned_seeds, 
                                                              additional_WD_factor=[WD_factor, WD_rate])

        if model == 'fiducial':
            rates_df.to_csv(f'../data/rates_optimisticCE_{save_path}.csv')
            if write_seeds:
                create_h5_file(f'../data/fc_stage_seeds/{write_seeds}_optimisticCE.h5', rates_dict)

            if 'rates_df_wd' in locals() and not rates_df_wd.empty and rates_df_wd.values.any():
                rates_df_wd.to_csv(f'../data/rates_optimisticCE_{save_path}_WD.csv')

        elif model == 'unstableCaseBB':
            rates_df.to_csv(f'../data/rates_unstableCaseBB_opt_{save_path}.csv')
            if write_seeds:
                create_h5_file(f'../data/fc_stage_seeds/{write_seeds}_unstableCaseBB_opt.h5', rates_dict)

            if 'rates_df_wd' in locals() and not rates_df_wd.empty and rates_df_wd.values.any():
                rates_df_wd.to_csv(f'../data/rates_unstableCaseBB_opt_{save_path}_WD.csv')

def main(file, save_path, CEE, Z, Z_max, m_min, m_max, MT1mask, MT2mask, selected_seeds=None, write_seeds=None):

    output_results(file, save_path, CEE, Z, Z_max, m_min, m_max, MT1mask, MT2mask, selected_seeds=selected_seeds, write_seeds=write_seeds)

    print("done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process data files and output rates.')
    parser.add_argument('file', type=str, help="File path.")
    parser.add_argument('--save_path', required=True, type=str, help='Path to save.')
    parser.add_argument('--CEE', action='store_true', help='Calculate rates with CEE.')

    # specific property ranges
    parser.add_argument('--Z', type=float, default=None, help='Calculate for specific metallicity.')
    parser.add_argument('--Z_max', type=float, default=None, help='Calculate for specific metallicity range.')
    parser.add_argument('--m_min', type=float, default=None, help='Calculate for specific mass range.')
    parser.add_argument('--m_max', type=float, default=None, help='Calculate for specific mass range.')


    # formation channel stuff
    parser.add_argument('--MT1mask', type=str, default=None, help='Mask for the first MT.')
    parser.add_argument('--MT2mask', type=str, default=None, help='Mask for the second MT.')


    parser.add_argument('--selected_seeds', type=list, default=None, help='Specific seeds for processing.')
    parser.add_argument('--write_seeds', type=str, default=None, help='Write seeds to h5 file.')
    
    args = parser.parse_args()
    main(args.file, args.save_path, args.CEE, args.Z, args.Z_max, args.m_min, args.m_max, args.MT1mask, args.MT2mask, args.selected_seeds, args.write_seeds)
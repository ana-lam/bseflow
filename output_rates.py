#!/usr/bin/env python3

import argparse
import os
import gzip
import json
import pandas as pd
from calculate_rates import calculate_simulation_rates, clean_buggy_systems
from formation_channels import identify_formation_channels
import h5py as h5
import numpy as np

def output_results(file, save_path, CEE=False, Z=None, MT1mask=None, MT2mask=None, selected_seeds=None):

    print("starting...")

    model = file.split("/")[-2] # record model from file path name

    cleaned_seeds, WD_factor, WD_rate = clean_buggy_systems(file, selected_seeds=None)

    print("calculating channels...")
    if MT1mask is not None:
        channels, masks = identify_formation_channels(file, selected_seeds=cleaned_seeds)
    
    print("calculating rates...")

    if MT1mask is None:
        rates_dict_wd, rates_df_wd = calculate_simulation_rates(file, formation_channel=None, formation_channel_2=None, selected_seeds=cleaned_seeds, white_dwarfs=True, additional_WD_factor=[WD_factor, WD_rate], CEE=CEE)
        rates_dict, rates_df = calculate_simulation_rates(file, formation_channel=None, formation_channel_2=None, selected_seeds=cleaned_seeds, additional_WD_factor=[WD_factor, WD_rate], CEE=CEE)
    elif MT2mask is None:
        rates_dict, rates_df = calculate_simulation_rates(file, formation_channel=masks[MT1mask], formation_channel_2=None, selected_seeds=cleaned_seeds, additional_WD_factor=[WD_factor, WD_rate])
    else:
        rates_dict, rates_df = calculate_simulation_rates(file, formation_channel=masks[MT1mask], formation_channel_2=(masks[MT1mask] & masks[MT2mask]), selected_seeds=cleaned_seeds, additional_WD_factor=[WD_factor, WD_rate])

    print("writing csv...")
    rates_df.to_csv(f'../data/rates_{model}_{save_path}.csv')
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
            rates_dict_wd, rates_df_wd = calculate_simulation_rates(file, formation_channel=None, formation_channel_2=None, optimistic_CE=True, selected_seeds=cleaned_seeds, white_dwarfs=True, additional_WD_factor=[WD_factor, WD_rate], CEE=CEE)
            rates_dict, rates_df = calculate_simulation_rates(file, formation_channel=None, formation_channel_2=None, selected_seeds=cleaned_seeds, additional_WD_factor=[WD_factor, WD_rate], CEE=CEE)
        elif MT2mask is None:
            rates_dict, rates_df = calculate_simulation_rates(file, formation_channel=masks[MT1mask], formation_channel_2=None, optimistic_CE=True, selected_seeds=cleaned_seeds, additional_WD_factor=[WD_factor, WD_rate])
        else:
            rates_dict, rates_df = calculate_simulation_rates(file, formation_channel=masks[MT1mask], formation_channel_2=(masks[MT1mask] & masks[MT2mask]), optimistic_CE=True, selected_seeds=cleaned_seeds, additional_WD_factor=[WD_factor, WD_rate])

        if model == 'fiducial':
            rates_df.to_csv(f'../data/rates_optimisticCE_{save_path}.csv')
            if 'rates_df_wd' in locals() and not rates_df_wd.empty and rates_df_wd.values.any():
                rates_df_wd.to_csv(f'../data/rates_optimisticCE_{save_path}_WD.csv')

        elif model == 'unstableCaseBB':
            rates_df.to_csv(f'../data/rates_unstableCaseBB_opt_{save_path}.csv')
            if 'rates_df_wd' in locals() and not rates_df_wd.empty and rates_df_wd.values.any():
                rates_df_wd.to_csv(f'../data/rates_unstableCaseBB_opt_{save_path}_WD.csv')

def main(file, save_path, CEE, Z, MT1mask, MT2mask):

    output_results(file, save_path, CEE, Z, MT1mask, MT2mask)

    print("done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process data files and output rates.')
    parser.add_argument('file', type=str, help="File path.")
    parser.add_argument('--save_path', required=True, type=str, help='Path to save.')
    parser.add_argument('--CEE', action='store_true', help='Calculate rates with CEE.')
    parser.add_argument('--Z', type=float, default=None, help='Calculate for specific metallicity')
    parser.add_argument('--MT1mask', type=str, default=None, help='Mask for the first MT.')
    parser.add_argument('--MT2mask', type=str, default=None, help='Mask for the second MT.')
    
    args = parser.parse_args()
    main(args.file, args.save_path, args.CEE, args.Z, args.MT1mask, args.MT2mask)
#!/usr/bin/env python3

import argparse
import os
import h5py as h5
import numpy as np

def create_h5_file(file, data):
    
    with h5.File(file, 'w') as hdf:
        for key, value in data.items():
            group = hdf.create_group(key)

            for sub_key, sub_value in value.items():
                if isinstance(sub_value, np.ndarray):
                    group.create_dataset(sub_key, data=sub_value)
                elif isinstance(sub_value, list):
                    group.create_dataset(sub_key, data=np.array(sub_value))
                else:
                    group.attrs[sub_key] = sub_value
    print(f'HDF5 file {file} created successfully.')





def grab_simulation_files(dir):

    model_files=[]
    for folder in os.listdir(dir):
        model_data_path = dir + "/" + folder
        try:
            files = os.listdir(model_data_path)
            if len(files)==1 and "Output" in files[0]:
                model_files.append(model_data_path+"/"+files[0])
        except:
            pass
    
    return model_files


def output_results(file, save_path, MT1mask, MT2mask=None, selected_seeds=None):

    model = file.split("/")[-2] # record model from file path name

    print("calculating channels...")
    channels, masks = identify_formation_channels(file, selected_seeds=selected_seeds)
    print("calculating rates...")
    if MT2mask is None:
        rates_dict, rates_df = calculate_simulation_rates(file, formation_channel=masks[MT1mask], formation_channel_2=None)
    else:
        rates_dict, rates_df = calculate_simulation_rates(file, formation_channel=masks[MT1mask], formation_channel_2=(masks[MT1mask] & masks[MT2mask]))

    print("writing csv...")
    rates_df.to_csv(f'../data/rates_{model}_{save_path}.csv')

    if model == 'fiducial' or model == 'unstableCaseBB':
        print("running calculation with optimistic CE...")
        print("calculating channels...")
        channels, masks = identify_formation_channels(file, selected_seeds=selected_seeds)
        print("calculating rates...")
        if MT2mask is None:
            rates_dict, rates_df = calculate_simulation_rates(file, formation_channel=masks[MT1mask], formation_channel_2=None, optimistic_CE=True)
        else:
            rates_dict, rates_df = calculate_simulation_rates(file, formation_channel=masks[MT1mask], formation_channel_2=(masks[MT1mask] & masks[MT2mask]), optimistic_CE=True)

        if model == 'fiducial':
            rates_df.to_csv(f'../data/rates_optimisticCE_{save_path}.csv')
            
            print("running calculation with optimistic CE...")
            print("calculating channels...")
            channels, masks = identify_formation_channels(file, selected_seeds=selected_seeds)
            print("calculating rates...")
            if MT2mask is None:
                rates_dict, rates_df = calculate_simulation_rates(file, formation_channel=masks[MT1mask], formation_channel_2=None, optimistic_CE=True)
            else:
                rates_dict, rates_df = calculate_simulation_rates(file, formation_channel=masks[MT1mask], formation_channel_2=(masks[MT1mask] & masks[MT2mask]), optimistic_CE=True)

        elif model == 'unstableCaseBB':
            rates_df.to_csv(f'../data/rates_unstableCaseBB_opt_{save_path}.csv')


def main(dir, save_path, MT1mask, MT2mask):
    
    data_files = grab_simulation_files(dir)

    for index, file in enumerate(tqdm(data_files, desc="Processing all model variations...")):
        output_results(file, save_path, MT1mask, MT2mask)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process data files and output rates.')
    parser.add_argument('dir', type=str, help="Directory of data files.")
    parser.add_argument('--save_path', required=True, type=str, help='Path to save.')
    parser.add_argument('--MT1mask', required=True, type=str, help='Mask for the first MT.')
    parser.add_argument('--MT2mask', type=str, default=None, help='Mask for the second MT.')
    
    args = parser.parse_args()
    main(args.dir, args.save_path, args.MT1mask, args.MT2mask)
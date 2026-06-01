import numpy as np
import pandas as pd
import h5py as h5
import multiprocessing as mp
import os
import re
import warnings

from bseflow.utils import in1d
from bseflow.data_dicts import model_variations
from bseflow.config import get_group, get_field, get_sn_type_code


def find_particular_files(directory, filename_pattern):
    """
    Find specific files in directory based on regex file name pattern.
    """

    pattern = re.compile(filename_pattern)

    matching_files = []
    for filename in os.listdir(directory):
        full_path = os.path.join(directory, filename)
        if os.path.isfile(full_path) and pattern.match(filename):
            matching_files.append([filename, full_path])

    return np.array(sorted(matching_files,key=lambda x: (x[0])))[:,1]


def process_file(args):
    """
    Wrapper function for multiprocessing.
    """
    file_path, group, fields, column_rename, selected_seeds, return_df, preview = args
    data = grab_h5_data(file_path, group, return_df, fields, selected_seeds=selected_seeds, preview=preview)

    if return_df:
        data.rename(columns=column_rename, inplace=True)
        return data
    
    else:
        return np.asarray(data)


def multiprocess_files(files_and_fields, selected_seeds=None, return_df=False, mp=False, num_processes=2, preview=False):
    """
    Executes h5 reading with or without multiprocessing.
    """
    
    if mp:
        with mp.Pool(num_processes) as pool:
            args_list = [(file_path, group, fields, column_rename) + (selected_seeds, return_df, preview) for file_path, group, fields, column_rename in files_and_fields]
            data = pool.map(process_file, args_list)

        if len(data) == 1:
            if len(data[0]) == 1 and not return_df:
                return data[0][0]
            else:
                return data[0]
    
    else:
        data = []
        for file_path, group, fields, column_rename in files_and_fields:
            data.append(process_file((file_path, group, fields, column_rename, selected_seeds, return_df, preview)))
        
        if len(data) == 1:
            if len(data[0]) == 1 and not return_df:
                        return data[0][0]
            else:
                return data[0]

    return data


def grab_h5_data(file_path, internal_group, return_df=False, fields = [], selected_seeds=None, preview=False):
    """
    Reads in specific data from h5 file
    group: specific file (e.g. 'systems', 'doubleCompactObjects', 'supernovae', etc)
    return_df: return data in dataframe if True else np.array
    fields: specific data columns to pull
    selected_seeds: only pull data for specific systems by seed/ID
    preview: print file.keys()
    """
    
    # translate internal group names to actual COMPAS HDF5 group names
    try:
        group = get_group(internal_group)
    except KeyError:
        # group not in config mapping - pass through as-is but warn user
        warnings.warn(f"Group '{internal_group}' not found in config mapping. Attempting to read group with this name directly from the HDF5 file.")
        group = internal_group

    # read h5py file
    with h5.File(file_path, 'r') as file:
        if group not in file:
            raise KeyError(
                "Group '{}' (mapped from '{}') not found in {}. "
                "Check compas_fields.groups in bseflow.yaml.".format(group, internal_group, file_path)
            )
 
        group_file = file[group]

        if preview:
            print(list(group_file.keys()))

        # we always want to grab SEED and set it as the index
        seed_fields = ['SEED', 'seed', 'randomSeed', 'm_randomSeed']
        seed_field = next((field for field in seed_fields if group_file.__contains__(field)), None)

        if seed_field:
            if return_df:
                data = pd.DataFrame(pd.DataFrame(group_file[seed_field][...].squeeze().astype(int), columns=['SEED'])).set_index('SEED')
            else:
                seeds = group_file[seed_field][...].squeeze().astype(int)
                if selected_seeds is not None:
                    selected_indices = np.in1d(seeds, selected_seeds)
                    seeds = seeds[selected_indices]
                data = [seeds]

        else:
            raise Exception("No seed field found in this HDF5 file.")
            
        # create columns/arrays for the specified fields
        # translate internal field names to actual COMPAS HDF5 field names
        sn_type_cache = None

        for internal_field in fields:
            try:
                actual_field = get_field(internal_field)
            except KeyError:
                # field not in config mapping - pass through as-is but warn user
                warnings.warn(f"Field '{internal_field}' not found in config mapping. Attempting to read field with this name directly from the HDF5 file.")
                actual_field = internal_field
            
            # special case for weights not present
            if actual_field is None:
                if internal_field == "weight":
                    n = len(seeds) if not return_df else len(data)
                    field_data = np.ones(n, dtype=float)
                else:
                    raise ValueError(
                        "Field '{}' maps to None in config but is not 'weight'. "
                        "Check compas_fields.fields in bseflow.yaml.".format(internal_field)
                    )
            elif internal_field == "Survived":
                raw = group_file[actual_field][...].squeeze()
                if selected_seeds is not None:
                    raw = raw[selected_indices]
                field_data = (~raw.astype(bool)).astype(int)  # invert survived to unbound
            elif internal_field in ('flagPISN', 'flagPPISN'):
                if sn_type_cache is None:
                    raw = group_file[actual_field][...].squeeze()
                    sn_type_cache = raw[selected_indices] if selected_seeds is not None else raw
                if internal_field == 'flagPISN':
                    field_data = (sn_type_cache == get_sn_type_code('PISN')).astype(int)
                elif internal_field == 'flagPPISN':
                    field_data = (sn_type_cache == get_sn_type_code('PPISN')).astype(int)
            # normal field handling
            else:
                if actual_field not in group_file:
                    raise KeyError(
                        "Field '{}' (mapped from '{}') not found in group '{}' of {}. "
                        "Check compas_fields.fields in bseflow.yaml.".format(actual_field, internal_field, group, file_path)
                    )
                field_data = group_file[actual_field][...].squeeze()
                if selected_seeds is not None:
                    field_data = field_data[selected_indices]
        
            if return_df:
                data[internal_field] = field_data
            else:
                data.append(field_data)
        
        # if we only want data for seeds of interest, filter
        if return_df and selected_seeds is not None:
            data = data[data.index.isin(selected_seeds)]
        
        return data


def sort_model_files(model_files, rates_specific=False):
    
    # rewrite this later to sort model files with any file path

    model_substrings = ['alpha0_1', 'alpha0_5', 'alpha10', 'alpha2_0', 'ccSNkick_100km_s',
                        'ccSNkick_30km_s', 'fiducial', 'massTransferEfficiencyFixed_0_25', 
                        'massTransferEfficiencyFixed_0_5', 'massTransferEfficiencyFixed_0_75',
                        'maxNSmass2_0', 'maxNSmass3_0', 'noBHkick', 'noPISN', 'optimisticCE',
                        'rapid', 'unstableCaseBB', 'unstableCaseBB_opt', 'wolf_rayet_multiplier_0_1',
                        'wolf_rayet_multiplier_5']

    model_sort = {}

    if rates_specific:
        for path in model_files:
            if "unstableCaseBB_opt" in path:
                model_name = "unstableCaseBB_opt"
            else:
                model_name = [sub for sub in model_substrings if sub in path][0]
            # model_name = re.sub(r'^rates_|_[^_]+$', '', path.split("/")[-1])
            if model_name in model_variations:
                model_sort[model_variations[model_name]['short']] = path
    else:
        for model_name in model_files:
            if model_name in model_variations:
                model_sort[model_variations[model_name]['short']] = model_name
    
    return [model_sort[key] for key in sorted(model_sort)]


def create_compact_h5(file, output_file, selected_seeds):
    
    with h5.File(file, 'r') as fdata:

    # create a new H5 file to store the filtered data
        with h5.File(output_file, 'w') as f_out:

            # loop over the groups in the file
            for group_name in fdata.keys():
                print("Working on", group_name, "...")
                group_file = fdata[group_name]

                f_out.create_group(group_name)

                seed_fields = ['SEED', 'seed', 'randomSeed', 'm_randomSeed']
                seed_field = next((field for field in seed_fields if group_file.__contains__(field)), None)

                seeds = group_file[seed_field][...].squeeze().astype(int)
                indices_to_keep = in1d(seeds, selected_seeds)

                # loop over datasets in group
                for dataset_name in group_file.keys():
                    dataset = group_file[dataset_name][...].squeeze()
                    if dataset.shape[0] == len(seeds):
                        filtered_data = dataset[indices_to_keep]
                    else:
                        filtered_data = dataset
                    
                    f_out[group_name].create_dataset(dataset_name, data=filtered_data)

    print("Filtered H5 file created successfully.")


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
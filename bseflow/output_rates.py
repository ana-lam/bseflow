import argparse
import os
from calculate_rates import BSESimulation
from formation_channels import identify_formation_channels
from file_processing import create_h5_file

def output_results(file, output_dir, save_path, CEE=False, Z=None, Z_max=None, m_min=None, m_max=None,
                   MT1mask=None, MT2mask=None, prop_filter=None, selected_seeds=None, write_seeds=None,
                   optimistic_CE=False, include_wds=False):
    
    print("starting rate calculation...")

    model = file.split("/")[-2]
    temp_sim = BSESimulation(file, selected_seeds=selected_seeds, CEE=CEE, include_wds=include_wds,
                        optimistic_CE=optimistic_CE)

    if Z:
        selected_seeds = temp_sim.specify_metallicity(Z, Z_max=Z_max)
    if m_min or m_max:
        selected_seeds = temp_sim.filter_by_property('systems', 'mass1', min_val=m_min, max_val=m_max)
    if prop_filter:
        if len(prop_filter) == 3:
            selected_seeds = temp_sim.filter_by_property(prop_filter[0], prop_filter[1], 
                                                        min_val=float(prop_filter[2]))
        elif len(prop_filter) == 4:
            selected_seeds = temp_sim.filter_by_property(prop_filter[0], prop_filter[1], 
                                                        min_val=float(prop_filter[2]), 
                                                        max_val=float(prop_filter[3]))
    
    sim = BSESimulation(file, selected_seeds=selected_seeds, CEE=CEE, include_wds=include_wds,
                        optimistic_CE=optimistic_CE, formation_channel=MT1mask, 
                        formation_channel_2=MT2mask)
            
    
    if not os.path.isdir(f'/mnt/home/alam1/ceph/data/{output_dir}'):
        os.makedirs(f'/mnt/home/alam1/ceph/data/{output_dir}', exist_ok=True)
    if write_seeds and not os.path.isdir(f'/mnt/home/alam1/ceph/data/{output_dir}/fc_stage_seeds'):
        os.makedirs(f'/mnt/home/alam1/ceph/data/{output_dir}/fc_stage_seeds', exist_ok=True)

    rates_dict, rates_df = sim.calculate_all_rates()

    suffix = "_WD" if include_wds else ""

    print("writing csv...")
    rates_df.to_csv(f'/mnt/home/alam1/ceph/data/{output_dir}/rates_{model}_{save_path}{suffix}.csv')
    if write_seeds:
        create_h5_file(f'/mnt/home/alam1/ceph/data/{output_dir}/fc_stage_seeds/{write_seeds}_{model}{suffix}.h5', rates_dict)

    if model =='fiducial' or model == 'unstableCaseBB':
        print("starting rate calculation with optimistic CE...")
        sim = BSESimulation(file, selected_seeds=selected_seeds, CEE=CEE, include_wds=include_wds,
                optimistic_CE=True, formation_channel=MT1mask, 
                formation_channel_2=MT2mask)
        print("writing csv...")
        if model == 'fiducial':
            rates_df.to_csv(f'/mnt/home/alam1/ceph/data/{output_dir}/rates_optimisticCE_{save_path}{suffix}.csv')
            if write_seeds:
                create_h5_file(f'/mnt/home/alam1/ceph/data/{output_dir}/fc_stage_seeds/{write_seeds}_optimisticCE{suffix}.h5', rates_dict)
        elif model == 'unstableCaseBB':
            rates_df.to_csv(f'/mnt/home/alam1/ceph/data/{output_dir}/rates_unstableCaseBB_opt_{save_path}{suffix}.csv')
            if write_seeds:
                create_h5_file(f'/mnt/home/alam1/ceph/data/{output_dir}/fc_stage_seeds/{write_seeds}_unstableCaseBB_opt{suffix}.h5', rates_dict)

def main(file, output_dir, save_path, CEE, Z, Z_max, m_min, m_max, MT1mask, MT2mask, prop_filter, 
         selected_seeds=None, write_seeds=None):

    output_results(file, output_dir, save_path, CEE, Z, Z_max, m_min, m_max, MT1mask, MT2mask, 
                   prop_filter, selected_seeds=selected_seeds, write_seeds=write_seeds)

    print("done!")            


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process data files and output rates.')
    parser.add_argument('file', type=str, help="File path.")
    parser.add_argument('--output_dir', type=str, default="data", help="Output directory.")
    parser.add_argument('--save_path', required=True, type=str, help='Path to save.')
    parser.add_argument('--CEE', action='store_true', help='Calculate rates with CEE.')

    # specific property ranges
    parser.add_argument('--Z', type=float, default=None, help='Calculate for specific metallicity.')
    parser.add_argument('--Z_max', type=float, default=None, help='Calculate for specific metallicity range.')
    parser.add_argument('--m_min', type=float, default=None, help='Calculate for specific mass range.')
    parser.add_argument('--m_max', type=float, default=None, help='Calculate for specific mass range.')

    # general property filter
    parser.add_argument("--filter", nargs='+',  # accept exactly 3 or 4 args
    help="Specify filters as two strings followed by one or two integers (e.g., group name min [max])",
    )

    # formation channel stuff
    parser.add_argument('--MT1mask', type=str, default=None, help='Mask for the first MT.')
    parser.add_argument('--MT2mask', type=str, default=None, help='Mask for the second MT.')


    parser.add_argument('--selected_seeds', type=list, default=None, help='Specific seeds for processing.')
    parser.add_argument('--write_seeds', type=str, default=None, help='Write seeds to h5 file.')
    
    args = parser.parse_args()
    main(args.file, args.output_dir, args.save_path, args.CEE, 
         args.Z, args.Z_max, args.m_min, args.m_max, args.MT1mask, 
         args.MT2mask, args.prop_filter, args.selected_seeds, args.write_seeds)                
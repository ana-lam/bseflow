import argparse
import os
from bseflow.calculate_rates import BSESimulation
from bseflow.file_processing import create_h5_file
from bseflow.config import get_rates_dir, get_seeds_subdir

def _resolve_dirs(output_dir=None, write_seeds=False):
    """
    Return (rates_dir, seeds_subdir) based on config and arguments.
    """
    rates_dir = output_dir if output_dir is not None else get_rates_dir()
    seeds_subdir = os.path.join(rates_dir, os.path.basename(get_seeds_subdir())) if write_seeds else None

    os.makedirs(rates_dir, exist_ok=True)
    if write_seeds:
        os.makedirs(seeds_subdir, exist_ok=True)
    
    return rates_dir, seeds_subdir


def output_results(file, output_dir=None, save_path=None, CEE=False, Z=None, Z_max=None, m_min=None, m_max=None,
                   MT1mask=None, MT2mask=None, prop_filter=None, selected_seeds=None, write_seeds=None,
                   optimistic_CE=False, include_wds=False):
    """
    Calculate rates for a COMPAS HDF5 file and write CSV (and optionally HDF5) outputs.

    Parameters:
    -----------
    file: str
        Path to the COMPAS HDF5 file.
    output_dir: str, optional
        Directory to write output files into. Defaults to bseflow.yaml value.
    save_path: str, optional
        Label used in output filenames.
    CEE: bool, optional
        Whether to calculate rates with common envelope evolution. Default False.
    Z: float, optional
        If specified, filter to this metallicity.
    Z_max: float, optional
        If specified with Z, filter to metallicity <= Z_max.
    m_min: float, optional
        If specified, filter to mass >= m_min.
    m_max: float, optional  
        If specified, filter to mass <= m_max.
    MT1mask: str, optional
        If specified, filter to systems with first MT matching this mask (formation channel specific).
    MT2mask: str, optional
        If specified, filter to systems with second MT matching this mask (formation channel specific).
    prop_filter: list of str, optional
        If specified, should be [group, property, min_val] or [group, property, min_val, max_val] to filter by any property.
    selected_seeds: list of int, optional
        If specified, only process these seeds.
    write_seeds: str, optional
        If specified, write seeds to an HDF5 file with this label in the filename.
    optimistic_CE: bool, optional
        Whether to also calculate rates with optimistic common envelope assumptions (only for certain models). Default False.
    include_wds: bool, optional
        Whether to include white dwarf systems in the rate calculations. Default False.
    """
    
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

    rates_dir, seeds_subdir = _resolve_dirs(output_dir, write_seeds)

    rates_dict, rates_df = sim.calculate_all_rates()

    suffix = "_WD" if include_wds else ""

    print("writing csv...")
    rates_df.to_csv(os.path.join(rates_dir, f"rates_{model}_{save_path}{suffix}.csv"))
    if write_seeds:
        create_h5_file(
            os.path.join(seeds_subdir, f"{write_seeds}_{model}{suffix}.h5"),
            rates_dict,
        )

    if model in ('fiducial', 'unstableCaseBB'):
        print("starting rate calculation with optimistic CE...")
        sim_opt = BSESimulation(file, selected_seeds=selected_seeds, CEE=CEE,
                                include_wds=include_wds, optimistic_CE=True,
                                formation_channel=MT1mask, formation_channel_2=MT2mask)
        rates_dict_opt, rates_df_opt = sim_opt.calculate_all_rates()
        print("writing csv...")
        if model == 'fiducial':
            rates_df_opt.to_csv(
                os.path.join(rates_dir, f"rates_optimisticCE_{save_path}{suffix}.csv"))
            if write_seeds:
                create_h5_file(
                    os.path.join(seeds_subdir, f"{write_seeds}_optimisticCE{suffix}.h5"),
                    rates_dict_opt,
                )
        elif model == 'unstableCaseBB':
            rates_df_opt.to_csv(
                os.path.join(rates_dir, f"rates_unstableCaseBB_opt_{save_path}{suffix}.csv"))
            if write_seeds:
                create_h5_file(
                    os.path.join(seeds_subdir, f"{write_seeds}_unstableCaseBB_opt{suffix}.h5"),
                    rates_dict_opt,
                )

def main(file, output_dir, save_path, CEE, Z, Z_max, m_min, m_max, MT1mask, MT2mask, prop_filter, 
         selected_seeds=None, write_seeds=None):

    output_results(file, output_dir, save_path, CEE, Z, Z_max, m_min, m_max, MT1mask, MT2mask, 
                   prop_filter, selected_seeds=selected_seeds, write_seeds=write_seeds)

    print("done!")            


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process COMPAS files and output rates.')
    parser.add_argument('file', type=str, help="Path to the COMPAS HDF5 file.")
    parser.add_argument('--output_dir', type=str, default="bseflow", help="Output directory.")
    parser.add_argument('--save_path', required=True, type=str, help='Label used in output filenames.')
    parser.add_argument('--CEE', action='store_true', help='Calculate rates with CEE.')

    # specific property ranges
    parser.add_argument('--Z', type=float, default=None, help='Calculate for specific metallicity.')
    parser.add_argument('--Z_max', type=float, default=None, help='Calculate for specific metallicity range.')
    parser.add_argument('--m_min', type=float, default=None, help='Calculate for specific mass range.')
    parser.add_argument('--m_max', type=float, default=None, help='Calculate for specific mass range.')

    # general property filter
    parser.add_argument("--prop_filter", nargs='+',  # accept exactly 3 or 4 args
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
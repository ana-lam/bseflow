import os
import glob
import tempfile
import pandas as pd

from bseflow.plotting.sankey import sankey_data_transform, plot_sankey

def render_from_rates(csv_path, title="", CEE=False, formation_channel=False):
    """
    Render a Sankey diagram from a CSV file containing rates data.
    
    Parameters:
    - csv_path: Path to the CSV file containing rates data.
    - title: Title of the Sankey diagram.
    - CEE: Boolean indicating if the data is for Common Envelope Evolution.
    - formation_channel: Boolean indicating if formation channels should be highlighted.
    
    Returns:
    - None
    """
    rates = pd.read_csv(csv_path, index_col=0)
    
    df = sankey_data_transform(rates, CEE=CEE, formation_channel=formation_channel)
    
    return plot_sankey(df, title=title, CEE=CEE,
                       formation_channel=formation_channel, return_fig=True)

def render_from_COMPAS_Output(h5_path, title="", CEE=False, formation_channel=False):
    """
    Render a Sankey diagram from a COMPAS H5 output file.
    
    Parameters:
    - h5_path: Path to the COMPAS H5 output file.
    - title: Title of the Sankey diagram.
    - CEE: Boolean indicating if the data is for Common Envelope Evolution.
    - formation_channel: Boolean indicating if formation channels should be highlighted.
    
    Returns:
    - None
    """
    # first run output rates
    from bseflow.output_rates import output_results
    
    with tempfile.TemporaryDirectory() as tmp:
        output_results(h5_path, save_path="web", output_dir=tmp, CEE=CEE)

        csvs = glob.glob(os.path.join(tmp, "*.csv"))
        if not csvs:
            raise RuntimeError("Reduction to rates failed. No CSVs were generated.")
        
        base = [c for c in csvs
                if "optimistic" not in os.path.basename(c)
                and "unstable" not in os.path.basename(c)]
        csv_path = (base or csvs)[0]
 
        return render_from_rates(
            csv_path, title=title,
            CEE=CEE, formation_channel=formation_channel,
        )
    
def render_sankey(path, **kwargs):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return render_from_rates(path, **kwargs)
    if ext in [".h5", ".hdf5"]:
        return render_from_COMPAS_Output(path, **kwargs)
    raise ValueError(f"Unsupported file extension: {ext}. Supported extensions are .csv, .h5, .hdf5.")
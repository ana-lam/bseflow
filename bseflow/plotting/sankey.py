import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.offline as pyo


def drake_table(df, factors):
    """
    Filter DataFrame to only keep relevant Drake-like rates.

    Parameters
    ----------
    df : `DataFrame`
        DataFrame with Drake-like rates used in Sankey and elsewhere for a specific model.

    factors : `list`
        Factors we want in the DataFrame.
    
    Returns
    -------
    df : `DataFrame`
        Filtered DataFrame with only the rates that describe DCO formation + merger.
    """

    initial_factor = df['wd_factor'].iloc[0]
    if "MR" in df.index.to_list():
        initial_factor+=df.loc['MR']['rates']
    if "PISN" in df.index.to_list():
        initial_factor+=df.loc['PISN']['rates']
    initial_rates_per_mass = df.loc['ZAMS']['rate_per_mass']

    df = df[df.index.isin(factors)]
    
    initial_row = pd.DataFrame({"rates": [initial_factor], "rate_per_mass":[initial_rates_per_mass]}, index=['f_initial'])

    df = pd.concat([initial_row, df], ignore_index=False)    

    return df


better_labels = {
    "ZAMS" : "ZAMS",
    "StellarMergerBeforeMT" : "Stellar Merger",
    "WDBeforeMT" : "White Dwarf",
    "MT1" : "Mass Transfer",
    "StellarMerger1" : "Stellar Merger",
    "WD1": "White Dwarf",
    "SN1" : "Supernova",
    "Disbound1" : "Disbound",
    "MT2" : "Mass Transfer",
    "StellarMerger2": "Stellar Merger",
    "WD2": "White Dwarf",
    "SN2" : "Supernova",
    "Disbound2" : "Disbound",
    "DCO" : "DCO Formation",
    "Merges" : r'Merges $< t_H$',
    "NoMerger" : r'Merges $> t_H$'
    }

def sankey_data_transform(df, CEE=False, formation_channel=False, custom_path=None):

    df = df[~df.index.str.contains("Survive")].copy()

    node_mapping = {
    "ZAMS" : "0",
    "StellarMergerBeforeMT" : "1",
    "WDBeforeMT" : "2",
    "MT1" : "3",
    "StellarMerger1" : "4",
    "WD1": "5",
    "SN1" : "6",
    "Disbound1" : "7",
    "MT2" : "8",
    "StellarMerger2": "9",
    "WD2": "10",
    "SN2" : "11",
    "Disbound2" : "12",
    "DCO" : "13",
    "Merges" : "14",
    "NoMerger" : "15"
    }

    CE_node_mapping = {
    "ZAMS" : "0",
    "StellarMergerBeforeMT" : "1",
    "WDBeforeMT" : "2",
    "CE1" : "3",
    "SMT1" : "4",
    "StellarMerger1" : "5",
    "WD1": "6",
    "SN1" : "7",
    "Disbound1" : "8",
    "CE2" : "9",
    "SMT2" : "10",
    "StellarMerger2": "11",
    "WD2": "12",
    "SN2" : "13",
    "Disbound2" : "14",
    "DCO" : "15",
    "Merges" : "16",
    "NoMerger" : "17"
    }

    fc_node_mapping = {
        "SN1" : "1",
        "StellarMerger1" : "2",
        "Disbound1" : "3",
        "MT2" : "4",
        "MT2other" : "5",
        "StellarMerger2" : "6",
        "SN2": "7",
        "Merges": "8",
        "NoMerger": "9"
    }

    df['source'] = ""
    df['target'] = ""

    for index, row in df.iterrows():
        if index == 'MR' or index == 'WD' or index == 'PISN':
            continue
        elif index == "ZAMS":
            df.loc[index, 'source'] = "0"
            df.loc[index, 'target'] = "0"
        else:
            source_key = index.split("_")[0]
            target_key = index.split("_")[1]
            
            df.loc[index, 'source'] = CE_node_mapping[source_key] if CEE is True else node_mapping[source_key]
            df.loc[index, 'target'] = CE_node_mapping[target_key] if CEE is True else node_mapping[target_key]

    trash_substrings = ['StellarMerger', 'WD', 'Disbound', 'NoMerger', 'MR', 'PISN']
    
    if formation_channel:
        trash_substrings = ['StellarMerger', 'WD', 'Disbound', 'NoMerger', 'MR', 'PISN', 'other']

    trash = [node_mapping[key] for key in node_mapping.keys() if any(map(key.__contains__, trash_substrings))]

    if CEE:
        trash = [CE_node_mapping[key] for key in CE_node_mapping.keys() if any(map(key.__contains__, trash_substrings))]

    df['color'] = df.apply(lambda row: "rgba(19, 121, 32, 0.73)" if row.target not in trash else "rgba(115, 115, 115, 0.4)", axis=1)

    if custom_path:
        df = df.replace("rgba(19, 121, 32, 0.73)", "rgba(220, 85, 0, 0.4)")
        for phase in custom_path:
            df.loc[phase, 'color'] = "rgba(19, 121, 32, 0.73)"

    if formation_channel:
         df.rename(index={'ZAMS_SN1': 'ZAMS_SN1other'}, inplace=True)
         df.loc[df.index.str.contains("ZAMS_SN1other"), 'color'] = "rgba(115, 115, 115, 0.4)"
         df.rename(index={'SN1_SN2': 'SN1_SN2other'}, inplace=True)
         df.loc[df.index.str.contains("SN1_SN2other"), 'color'] = "rgba(115, 115, 115, 0.4)"
    df['values'] = df[df.columns[0]] * 100

    if formation_channel:
        df.loc[df.index.str.contains("MT1_SN1"), 'target'] = "6"
        df.loc[df.index.str.contains("SN1_Disbound1"), 'source'] = "6"
        df.loc[df.index.str.contains("SN1_Disbound1"), 'target'] = "7"
        df.loc[df.index.str.contains("SN1_MT2"), 'source'] = "6"
        df.loc[df.index.str.contains("SN1_MT2"), 'target'] = "8"
        df.loc[df.index.str.contains("MT2_StellarMerger2"), 'source'] = "8"
        df.loc[df.index.str.contains("MT2_StellarMerger2"), 'target'] = "9"
        df.loc[df.index.str.contains("MT2_SN2"), 'source'] = "9"
        df.loc[df.index.str.contains("MT2_SN2"), 'target'] = "10"
        df.loc[df.index.str.contains("SN1_SN2other"), 'source'] = "6"
        df.loc[df.index.str.contains("SN1_SN2other"), 'target'] = "15"

    df = df.iloc[4: , :]

    return df

def plot_sankey(df, title="", CEE=False, formation_channel=False, save_path=None, custom_trash=None):

    node_mapping = {
    "ZAMS" : "0",
    "StellarMergerBeforeMT" : "1",
    "WDBeforeMT" : "2",
    "MT1" : "3",
    "StellarMerger1" : "4",
    "WD1": "5",
    "SN1" : "6",
    "Disbound1" : "7",
    "MT2" : "8",
    "StellarMerger2": "9",
    "WD2": "10",
    "SN2" : "11",
    "Disbound2" : "12",
    "DCO" : "13",
    "Merges" : "14",
    "NoMerger" : "15"
    }
    
    CE_node_mapping = {
        "ZAMS" : "0",
        "StellarMergerBeforeMT" : "1",
        "WDBeforeMT" : "2",
        "CE1" : "3",
        "SMT1" : "4",
        "StellarMerger1" : "5",
        "WD1": "6",
        "SN1" : "7",
        "Disbound1" : "8",
        "CE2" : "9",
        "SMT2" : "10",
        "StellarMerger2": "11",
        "WD2": "12",
        "SN2" : "13",
        "Disbound2" : "14",
        "DCO" : "15",
        "Merges" : "16",
        "NoMerger" : "17"
        }

    better_labels = {
    "ZAMS" : "ZAMS",
    "StellarMergerBeforeMT" : "Stellar Merger",
    "WDBeforeMT" : "White Dwarf",
    "MT1" : "Mass Transfer",
    "CE1" : "CE",
    "SMT1" : "Stable MT",
    "StellarMerger1" : "Stellar Merger",
    "WD1": "White Dwarf",
    "SN1" : "Supernova",
    "Disbound1" : "Unbound",
    "MT2" : "Mass Transfer",
    "CE2" : "CE",
    "SMT2" : "Stable MT",
    "StellarMerger2": "Stellar Merger",
    "WD2": "White Dwarf",
    "SN2" : "Supernova",
    "Disbound2" : "Unbound",
    "DCO" : "DCO Formation",
    "Merges" : r'Merges $< t_H$',
    "NoMerger" : r'Merges $> t_H$'
    }

    trash_substrings = ['StellarMerger', 'WD', 'Disbound', 'NoMerger']
    if formation_channel:
        trash_substrings = ['StellarMerger', 'WD', 'Disbound', 'NoMerger', 'other']

    trash = [node_mapping[key] for key in node_mapping.keys() if any(map(key.__contains__, trash_substrings))]

    if CEE:
        trash = [CE_node_mapping[key] for key in CE_node_mapping.keys() if any(map(key.__contains__, trash_substrings))]

    labels = [better_labels[key].replace(r'$< t_H$', '<t<sub>H</sub>').replace(r'$> t_H$', '>t<sub>H</sub>') for key in node_mapping.keys()]
    if CEE:
        labels = [better_labels[key].replace(r'$< t_H$', '<t<sub>H</sub>').replace(r'$> t_H$', '>t<sub>H</sub>') for key in CE_node_mapping.keys()]
    # labels = [key.replace("1", "").replace("2", "").replace("3", "") for key in node_mapping.keys()]
    
    node_colors = ["rgba(19, 121, 32, 0.73)" if value not in trash
               else "rgba(115, 115, 115, 0.4)" for value in node_mapping.values()]
    if CEE:
        node_colors = ["rgba(19, 121, 32, 0.73)" if value not in trash
               else "rgba(115, 115, 115, 0.4)" for value in CE_node_mapping.values()]
    if custom_trash:
        for phase in custom_trash:  
            phase_index = list(CE_node_mapping.keys()).index(phase)
            node_colors[phase_index] = "rgba(220, 85, 0, 0.4)"

    fig = go.Figure(data=[go.Sankey(
    valueformat='.3r',
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=.5),
        label=labels,
        color=node_colors,
        hovertemplate='%{label}: %{value}%<extra></extra>'
    ),
    link=dict(
        source=df['source'],  # indices correspond to labels, eg 'MT1' indices are 3 and 4
        target=df['target'],
        value=df['values'],
        color=df['color'],
        hovertemplate='%{source.label} to %{target.label}: %{value}%<extra></extra>'
    ))])

    fig.show()

    fig.update_layout(title_text=f'{title}', font=dict(size=20))
    if save_path:
        fig.write_html(f'/mnt/home/alam1/paper_figures/sankey_htmls/{save_path}')
    else:
        fig.write_html(f'/mnt/home/alam1/paper_figures/sankey_htmls/{df.columns[0].split("_")[0]}.html')
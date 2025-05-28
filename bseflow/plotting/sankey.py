import pandas as pd
import plotly.graph_objects as go


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
        "StellarMerger1" : "1",
        "WD1": "5",
        "SN1" : "6",
        "Disbound1" : "7",
        "MT2" : "8",
        "StellarMerger2": "1",
        "WD2": "10",
        "SN2" : "11",
        "Disbound2" : "7",
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
        "StellarMerger1" : "1",
        "WD1": "6",
        "SN1" : "7",
        "Disbound1" : "8",
        "CE2" : "9",
        "SMT2" : "10",
        "StellarMerger2": "1",
        "WD2": "12",
        "SN2" : "13",
        "Disbound2" : "8",
        "DCO" : "15",
        "Merges" : "16",
        "NoMerger" : "17"
        }

    fc_node_mapping = {
        "MT1": "0",
        "SN1" : "1",
        "StellarMerger1" : "2",
        "Disbound1" : "3",
        "MT2" : "4",
        "MT2other" : "5",
        "StellarMerger2" : "2",
        "SN2": "7",
        "Disbound2": "3",
        "DCO" : "8",
        "Merges": "9",
        "NoMerger": "10"
    }

    df['source'] = ""
    df['target'] = ""

    for index, row in df.iterrows():
        if formation_channel:
            if index =='MR' or index == 'WD' or index == "PISN" or index == "ZAMS" \
                or index == "ZAMS_StellarMergerBeforeMT" or index == "ZAMS_MT1other" or index =='ZAMS_SN1':
                continue
            elif index == 'ZAMS_MT1':
                df.loc[index, 'source'] = "0"
                df.loc[index, 'target'] = "0"
            else:
                source_key = index.split("_")[0]
                target_key = index.split("_")[1]

                df.loc[index, 'source'] = fc_node_mapping[source_key]
                df.loc[index, 'target'] = fc_node_mapping[target_key]
        else:
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
        trash_substrings = ['StellarMerger', 'WD', 'Disbound', 'NoMerger', 'other']
        trash = [fc_node_mapping[key] for key in fc_node_mapping.keys() if any(map(key.__contains__, trash_substrings))]
    elif CEE:
        trash = [CE_node_mapping[key] for key in CE_node_mapping.keys() if any(map(key.__contains__, trash_substrings))]
    else:
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
         df.rename(index={'SN1_SN2': 'SN1_SN2other'}, inplace=True)
         df.loc['SN1_SN2other', 'target'] = 6
         df.loc[df.index.str.contains("other"), 'color'] = "rgba(220, 85, 0, 0.4)"
    df['values'] = df[df.columns[0]] * 100

    df = df.iloc[4: , :]

    if formation_channel:
        df = df[~df.index.str.contains('ZAMS')]
        renormalize_value = df.iloc[1]['values'] if df.iloc[0]['values'] == 0 else df.iloc[0]['values']
        df['values'] = df['values']/renormalize_value * 100

    return df

def plot_sankey(df, title="", CEE=False, formation_channel=False, save_path=None, custom_path_labels=None, custom_trash=None):

    node_mapping = {
        "ZAMS" : "0",
        "StellarMergerBeforeMT" : "1",
        "WDBeforeMT" : "2",
        "MT1" : "3",
        "StellarMerger1" : "1",
        "WD1": "5",
        "SN1" : "6",
        "Disbound1" : "7",
        "MT2" : "8",
        "StellarMerger2": "1",
        "WD2": "10",
        "SN2" : "11",
        "Disbound2" : "7",
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
        "StellarMerger1" : "1",
        "WD1": "6",
        "SN1" : "7",
        "Disbound1" : "8",
        "CE2" : "9",
        "SMT2" : "10",
        "StellarMerger2": "1",
        "WD2": "12",
        "SN2" : "13",
        "Disbound2" : "8",
        "DCO" : "15",
        "Merges" : "16",
        "NoMerger" : "17"
        }
    
    fc_node_mapping = {
        "MT1": "0",
        "SN1" : "1",
        "StellarMerger1" : "2",
        "Disbound1" : "3",
        "MT2" : "4",
        "MT2other" : "5",
        "SN2other" : "6",
        "StellarMerger2" : "2",
        "SN2": "7",
        "Disbound2": "3",
        "DCO" : "8",
        "Merges": "9",
        "NoMerger": "10"
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
    
    fc_better_labels = {
        "MT1": f"{custom_path_labels[0]}",
        "SN1" : "Supernova",
        "StellarMerger1" : "Stellar Merger",
        "Disbound1" : "Unbound",
        "MT2" : f"{custom_path_labels[1]}",
        "MT2other" : f"{custom_path_labels[2]}",
        "SN2other" : "Supernova",
        "StellarMerger2" : "Stellar Merger",
        "SN2": "Supernova",
        "Disbound2": "Unbound",
        "DCO" : "DCO Formation",
        "Merges" : r'Merges $< t_H$',
        "NoMerger" : r'Merges $> t_H$'
        }

    trash_substrings = ['StellarMerger', 'WD', 'Disbound', 'NoMerger']
    if formation_channel:
        trash_substrings = ['StellarMerger', 'WD', 'Disbound', 'NoMerger']
        trash = [fc_node_mapping[key] for key in fc_node_mapping.keys() if any(map(key.__contains__, trash_substrings))]
    
    elif CEE:
        trash = [CE_node_mapping[key] for key in CE_node_mapping.keys() if any(map(key.__contains__, trash_substrings))]
    
    else:
        trash = [node_mapping[key] for key in node_mapping.keys() if any(map(key.__contains__, trash_substrings))]

    if formation_channel:

        labels = [fc_better_labels[key].replace(r'$< t_H$', '<t<sub>H</sub>').replace(r'$> t_H$', '>t<sub>H</sub>') for key in fc_node_mapping.keys()]

        fc_node_mapping_int = {k: int(v) for k, v in fc_node_mapping.items()}

        max_index = max(fc_node_mapping_int.values())
        labels = [""] * (max_index + 1)

        for phase, idx in fc_node_mapping_int.items():
            label_text = fc_better_labels.get(phase, phase)
            label_text = label_text.replace(r'$< t_H$', '<t<sub>H</sub>').replace(r'$> t_H$', '>t<sub>H</sub>')
            labels[idx] = label_text

    elif CEE:
        labels = [better_labels[key].replace(r'$< t_H$', '<t<sub>H</sub>').replace(r'$> t_H$', '>t<sub>H</sub>') for key in CE_node_mapping.keys()]
    else:
        labels = [better_labels[key].replace(r'$< t_H$', '<t<sub>H</sub>').replace(r'$> t_H$', '>t<sub>H</sub>') for key in node_mapping.keys()]
    
    if formation_channel:
        node_colors = ["rgba(19, 121, 32, 0.73)",
                       "rgba(19, 121, 32, 0.73)",
                       "rgba(115, 115, 115, 0.4)",
                       "rgba(115, 115, 115, 0.4)",
                       "rgba(19, 121, 32, 0.73)",
                       "rgba(220, 85, 0, 0.4)",
                       "rgba(220, 85, 0, 0.4)",
                       "rgba(19, 121, 32, 0.73)",
                       "rgba(19, 121, 32, 0.73)",
                       "rgba(19, 121, 32, 0.73)",
                       "rgba(115, 115, 115, 0.4)",]

    elif CEE:
        node_colors = ["rgba(19, 121, 32, 0.73)" if value not in trash
               else "rgba(115, 115, 115, 0.4)" for value in CE_node_mapping.values()]
    else:
        node_colors = ["rgba(19, 121, 32, 0.73)" if value not in trash
               else "rgba(115, 115, 115, 0.4)" for value in node_mapping.values()]

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
        fig.write_html(f'/mnt/home/alam1/bseflow/sankey_htmls/{save_path}')
    else:
        fig.write_html(f'/mnt/home/alam1/bseflow/sankey_htmls/{df.columns[0].split("_")[0]}.html')
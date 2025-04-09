import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator
import numpy as np
from matplotlib.font_manager import FontProperties
import re


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fs=20
params = {'legend.fontsize': fs,
          'axes.labelsize': fs,
          'font.weight': 'bold',
          'axes.labelweight': 'bold',
          'xtick.labelsize': 0.7*fs,
          'ytick.labelsize': 0.7*fs}
plt.rcParams.update(params)
plt.rcParams['pdf.fonttype'] = 42

plt.rcParams.update({
    'axes.edgecolor': '#545454',  # Dark gray for axes borders
    'xtick.color': '#545454',     # Dark gray for x-tick labels
    'ytick.color': '#545454',     # Dark gray for y-tick labels
    'axes.labelcolor': 'black', # Dark gray for axes labels
    'text.color': 'black',      # Dark gray for text
    'axes.titlecolor': 'black', # Dark gray for title
    'axes.linewidth': 1.5,        # Thicker axes borders
    'xtick.major.width': 1.0,     # Thicker major x-ticks
    'ytick.major.width': 1.0,     # Thicker major y-ticks
    'xtick.minor.width': 1,     # Thicker minor x-ticks
    'ytick.minor.width': 1      # Thicker minor y-ticks
})

# lighten color for plotting purposes
def lighten_color(hex_color, amount=0.8):
    rgb = mcolors.hex2color(hex_color)

    lightened_rgb = [(1 - amount) * channel + amount for channel in rgb]

    lightened_hex = mcolors.to_hex(lightened_rgb)
    
    return lightened_hex


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

def survival_plot(df, title, labels, save_path=None, log=False):

    fig, ax = plt.subplots(1,1, figsize=(10,6))
    for i, df in enumerate(df):
        for substring in trash_substrings:
            indices_to_remove = [index for index in df.index if substring in index]
            df = df.drop(indices_to_remove)

        index_labels = [index.replace('_', ' to ') for index in df.index]

        ax.plot(index_labels, df.iloc[:, 0], drawstyle='steps-post', color=colors[i], linewidth=3, label=f'{labels[i]}', alpha=1)
  
    # ax = layoutAxes(ax, nameX='Critical Events', nameY='Survival Rate', setMinor=False, labelpad=10, fontsize=12, labelSizeMajor=18)  

    ax.set_xlabel('Critical Events', fontsize=16, fontweight='bold')
    # plt.ylabel("Survival Rate")
    ax.set_title(f'{title}', fontsize=20)
    ax.grid(True)
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -.2), fancybox=True, ncol=len(labels), fontsize=15)

    for text in legend.get_texts():
        text.set_fontweight('bold')

    if log:
        ax.set_yscale('log')
        ax.grid(False)

    if save_path:
        fig.save_fig(f'{save_path}')
    plt.show()


def factors_plot(ax, df, fc_color, x_label=None, y_label=None, title=None, fontsize=14, legend=False, MT2=True, mt1='MT1', mt2='MT2',
                 other_MT_1='MT1 Other', other_MT_2='MT2 Other'):

    rel_rates = df['rel_rates']
    survive_string_format = 'survive\ '

    f_MT1 = {fr'$f_{{\mathbf{{{mt1}}}}}$': rel_rates['ZAMS_MT1'], 'MT1 Other': rel_rates['ZAMS_MT1other'], 'SN': rel_rates['ZAMS_SN1'], 'Stellar Merger': rel_rates['ZAMS_StellarMergerBeforeMT']}
    f_MT1_survive = {fr'$f_{{\mathbf{{{survive_string_format + mt1}}}}}$': rel_rates['MT1_SN1'], 'Stellar Merger': rel_rates['MT1_StellarMerger1']}
    f_SN1_survive = {fr'$f_{{\mathbf{{survive\ SN1}}}}$': rel_rates['SN1_Survive1'], 'Disbound': rel_rates['SN1_Disbound1']}
    if MT2 is True:
        f_MT2 = {fr'$f_{{\mathbf{{{mt2}}}}}$': rel_rates['SN1_MT2'], 'MT2 Other': rel_rates['SN1_MT2other'], 'SN': rel_rates['SN1_SN2']}
    else:
        f_MT2 = {'MT2': rel_rates['SN1_MT2'], 'SN': rel_rates['SN1_SN2']}
    f_MT2_survive = {fr'$f_{{\mathbf{{{survive_string_format + mt2}}}}}$': rel_rates['MT2_SN2'], 'Stellar Merger': rel_rates['MT2_StellarMerger2']}
    f_SN2_survive = {fr'$f_{{\mathbf{{survive\ SN2}}}}$': rel_rates['SN2_Survive2'], 'Disbound': rel_rates['SN2_Disbound2']}
    f_DCO_merges = {fr'$f_{{\mathbf{{merge}}}}$': rel_rates['DCO_Merges'], 'No Merger': rel_rates['DCO_NoMerger']}

    all_factors = [f_MT1, f_MT1_survive, f_SN1_survive, f_MT2, f_MT2_survive,f_SN2_survive, f_DCO_merges]

    for factor in all_factors:
        labels = list(factor.keys())
        vals = list(factor.values())
        bottom = 0

        for index, label in enumerate(labels):
            if index == 0:
                color = '#427e29'
                hatch = None
            elif label == 'MT1 Other':
                color = '#badcf8'
                hatch = '.'
                label = f'{other_MT_1}'
            elif label == 'MT2 Other':
                color = '#f0ba4e'
                hatch = 'O.'
                label = f'{other_MT_2}'
            elif label == 'Stellar Merger':
                color = '#bc4c3a'
                hatch = '//'
            elif label == 'Disbound':
                color = '#9353c0'
                hatch = 'x'
            elif label == 'SN':
                color = '#1d8bd7'
                hatch = '\\'
            elif label == 'No Merger':
                color = '#c76639'
                hatch = 'o'

            ax.bar(list(factor.keys())[0], vals[index], bottom=bottom, color=color, edgecolor='#545454', width=1, hatch=hatch, linewidth=2)
            bottom += vals[index] 

    hatch_labels = {
    fr'$\textbf{{{other_MT_1}}}$': {'hatch': '.', 'facecolor': '#badcf8'},
    fr'$\textbf{{{other_MT_2}}}$': {'hatch': 'O.', 'facecolor': '#f0ba4e'},
    fr'$\textbf{{Stellar Merger}}$': {'hatch': '//', 'facecolor': '#dab7f5'},
    fr'$\textbf{{Disbound}}$': {'hatch': 'x', 'facecolor': '#9353c0'},
    fr'$\textbf{{Supernova}}$': {'hatch': '\\', 'facecolor': '#1d8bd7'},
    fr'$\textbf{{No Merger}}$ ($\mathbf{{t > t_{{H}}}}$)': {'hatch': 'o', 'facecolor': '#c76639'}
    }

    handles = [
    mpatches.Patch(facecolor=attrs['facecolor'], edgecolor='black', hatch=attrs['hatch'], label=label)
    for label, attrs in hatch_labels.items()
    ]

    if legend:
        legend1 = ax.legend(handles=handles, bbox_to_anchor=(0.5, -0.35), loc='upper center', ncol=3, fontsize=fs+10)
        for text in legend1.get_texts():
            text.set_color('#545454')

    ax.set_ylim(0, 1)
    ax.set_yticks([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
    ax.set_yticklabels([r'\textbf{0.0}', r'\textbf{0.2}', r'\textbf{0.4}', r'\textbf{0.6}', r'\textbf{0.8}', r'\textbf{1.0}'], fontsize=fs+15, fontweight='heavy')

    for y in np.arange(0, 1.1, 0.1):
        ax.axhline(y, color="#545454", linestyle="-", lw=1, alpha=0.3)

    ax.set_xticks(range(len(all_factors)))
    ax.set_xticklabels([list(f.keys())[0] for f in all_factors], rotation=45, ha='right', fontsize=fs+25, fontweight='heavy')

    ax.set_xlim(-1.5, len(all_factors))

    if y_label is not None:
        ax.set_ylabel('Relative Rate', fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize+2)

    ax.tick_params(axis='both', which='major', length=10, width=8)
    ax.tick_params(axis='both', which='minor', length=5, width=1)


def plot_model_rates(rates, ax, ax2=None, MT2=True, log=False, fs=fs, x_top_labels=False, x_bottom_labels=False):

    # load in model labels & factor labels formatting
    with open('bseFlow/data_dicts/modelVariations.json') as f:
        variations = json.load(f)
    with open('bseFlow/data_dicts/drakeFactors.json') as f:
        drake_factors = json.load(f)

    # store disparity from fiducial value for ax2
    disparities = []

    fiducial_df = drake_table(pd.read_csv(rates[0], index_col=0), MT2=MT2)

    for index, factor in enumerate(fiducial_df.index.to_list()[1:]):
        model_values = []
        model_labels = []
        for model in rates:
            df = pd.read_csv(model, index_col=0)
            # grab model name from file path
            model = re.sub(r'^rates_|_[^_]+$', '', model.split("/")[-1])
            model_labels.append(model)
            df = drake_table(df, MT2=MT2)
            model_values.append(df['Drake Factors (relative)'].loc[factor])
            if model == 'fiducial':
                fiducial_value = df['Drake Factors (relative)'].loc[factor]
        model_xlabels = [variations[label]['short'] for label in model_labels]
        model_xlabels_long = [variations[label]['med'] for label in model_labels]

        disparities.append([fiducial_value - min(model_values), max(model_values) - fiducial_value])

        # markers
        ax.plot(model_xlabels, model_values, color=drake_factors[factor]['color'], marker='.', linewidth='4', markersize=18, markeredgecolor='k', markeredgewidth=0.8, alpha=0.6)
        
        # lines
        for i, model_value in enumerate(model_values):
            ax.plot(model_xlabels[i], model_value, marker='.', markersize=20, color=drake_factors[factor]['color'], markeredgecolor=drake_factors[factor]['color'], markeredgewidth=0.8)
    
    # fiducial values
    for index, row in fiducial_df[1:].iterrows():
        ax.axhline(y=row['Drake Factors (relative)'], color = drake_factors[index]['color'], linestyle=(0, (1, 1)), linewidth=2, label=drake_factors[index]['name'])

    # specify x ticks
    ax.tick_params(axis='x', which='both', length=10)
    ax.set_xlim(-1, len(model_xlabels))

    # set bottom x axis labels
    ax.set_xticklabels([])
    if x_bottom_labels:
        bold_labels = [r'\textbf{' + label + '}' for label in model_xlabels]
        ax.set_xticklabels(bold_labels, fontsize=fs*.8)
        ax.set_xlabel(r'\textbf{binary population synthesis model} $\mu$', fontsize=fs*.8)

    # set top x axis labels
    ax_top = ax.twiny()
    ax_top.set_xticks(range(len(model_xlabels)))
    ax_top.set_xlim(ax.get_xlim())   
    ax_top.set_xticklabels([])
    if x_top_labels:
        ax_top.set_xticklabels(model_xlabels_long, rotation=45, fontsize=0.5*fs)  
        ax_top.set_xlim(ax.get_xlim()) 

    # specify y ticks
    ax.set_ylim(-0.05, 1.05)

    if log:
        ax.set_yscale('log')
        ax.set_ylim(0.01, 1.05)
        ax.set_yticks([1, 0.1])
        ax.set_yticklabels([r'$$\mathbf{10^{0}}$$', r'$$\mathbf{10^{-1}}$$'], fontsize=fs*.8)
    else:
        pass
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_yticklabels([r'$$\mathbf{0.0}$$', r'$$\mathbf{0.2}$$', r'$$\mathbf{0.4}$$', r'$$\mathbf{0.6}$$', r'$$\mathbf{0.8}$$', r'$$\mathbf{1.0}$$'], fontsize=fs*.8)

    # ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.set_ylabel(r'\textbf{fraction}', fontsize=fs*.8)
    ax.tick_params(axis='y', which='major', length=12)
    ax.tick_params(axis='y', which='minor', length=6)

    ax.grid(axis='y', which='minor', linestyle='dotted')
    ax.grid(axis='y', which='major', linestyle='-')

    # right side uncertainty/disparity plot
    if ax2:
        fiducial_values = fiducial_df['Drake Factors (relative)'][1:].values
        factors = fiducial_df.index.to_list()[1:]

        for i, (factor, value, err) in enumerate(zip(factors, fiducial_values, disparities)):
            ax2.errorbar([drake_factors[factor]['name']], [value], yerr=np.array([[err[0]], [err[1]]]), fmt='o', markersize=10, 
                color = drake_factors[factor]['color'], ecolor=drake_factors[factor]['color'], capsize=10,
                elinewidth=2, capthick=2)
            
        # specify x ticks
        ax2.set_xticklabels([drake_factors[f]['name'] for f in factors], rotation=45) 
        ax2.set_xticklabels([]) 
        ax2.set_xticks([])
        ax.set_xlabel(r'\textbf{factor} $f$', fontsize=fs*.8)
        ax2.tick_params(axis='x', which='both', length=10)

        # specify y ticks
        ax2.set_ylim(-0.05, 1.05)
        if log:
            ax2.set_yscale('log')
            ax2.set_ylim(0.01, 1.05)

        ax2.set_yticklabels([]) 
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        
        ax2.yaxis.set_major_locator(MultipleLocator(0.2))
        ax2.yaxis.set_minor_locator(MultipleLocator(0.1))

        ax2.tick_params(axis='y', which='major', length=12)
        ax2.tick_params(axis='y', which='minor', length=6)
        
        ax2.grid(axis='y', which='minor', linestyle='dotted')
        ax2.grid(axis='y', which='major', linestyle='-')


        return ax, ax2
    else:
        return ax
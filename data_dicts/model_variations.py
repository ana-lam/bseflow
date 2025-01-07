model_variations = {
    "fiducial": {
        "file": "fiducial",
        "name": "Fiducial",
        "short": "A",
        "med": "fiducial",
        "color": "gray",
        "med_bold": r"\textbf{fiducial}",
        "med_bold2": r"\textbf{fiducial}",
        "marker": "*"
    },
    "massTransferEfficiencyFixed_0_25": {
        "file": "massTransferEfficiencyFixed_0_25",
        "name": r"Fixed mass transfer efficiency of $\beta=0.25$",
        "short": "B",
        "med": r"$\beta=0.25$",
        "color": "#dcb654",
        "med_bold": r"$\boldsymbol{\beta = 0.25}$",
        "med_bold2": r"$\boldsymbol{\beta = 0.25}$",
        "marker": "p"
    },
    "massTransferEfficiencyFixed_0_5": {
        "file": "massTransferEfficiencyFixed_0_5",
        "name": r"Fixed mass transfer efficiency of $\beta=0.5$",
        "short": "C",
        "med": r"$\beta=0.5$",
        "color": "#edc247",
        "med_bold": r"$\boldsymbol{\beta = 0.5}$",
        "med_bold2": r"$\boldsymbol{\beta = 0.5}$",
        "marker":"h"
    },
    "massTransferEfficiencyFixed_0_75": {
        "file": "massTransferEfficiencyFixed_0_75",
        "name": r"Fixed mass transfer efficiency of $\beta=0.75$",
        "short": "D",
        "med": r"$\beta=0.75$",
        "color": "#ffcd36",
        "med_bold": r"$\boldsymbol{\beta = 0.75}$",
        "med_bold2": r"$\boldsymbol{\beta = 0.75}$",
        "marker":"H"
    },
    "unstableCaseBB": {
        "file": "unstableCaseBB",
        "name": "Case BB mass transfer always unstable",
        "short": "E",
        "med": r"\shortstack{unstable \\ case BB}",  
        "color": "#db7814",
        "med_bold": r"\shortstack{\textbf{unstable} \\ \textbf{case BB}}",
        "med_bold2": r"\shortstack{\textbf{unstable} \textbf{case BB}}",
        "marker": "x"
    },
    "unstableCaseBB_opt": {
        "file": "unstableCaseBB_opt",
        "name": "E + K",
        "short": "F",
        "med": r"E + K",
        "color": "#e6600e",
        "med_bold": r"\textbf{E + K}",
        "med_bold2": r"\textbf{E + K}",
        "marker": "X"
    },
    "alpha0_1": {
        "file": "alpha0_1",
        "name": r"CE efficiency parameter of $\alpha=0.1$",
        "short": "G",
        "med": r"$\alpha_{\rm CE}=0.1$",
        "color": "#f8927d",
        "med_bold": r"$\boldsymbol{\alpha_{\rm CE}=0.1}$",
        "med_bold2": r"$\boldsymbol{\alpha_{\rm CE}=0.1}$",
        "marker": "v"
    },
    "alpha0_5": {
        "file": "alpha0_5",
        "name": r"CE efficiency parameter of $\alpha=0.5$",
        "short": "H",
        "med": r"$\alpha_{\rm CE}=0.5$",
        "color": "#f1755f",
        "med_bold": r"$\boldsymbol{\alpha_{\rm CE}=0.5}$",
        "med_bold2": r"$\boldsymbol{\alpha_{\rm CE}=0.5}$",
        "marker": "<"
    },
    "alpha2_0": {
        "file": "alpha2_0",
        "name": r"CE efficiency parameter of $\alpha=2.0$",
        "short": "I",
        "med": r"$\alpha_{\rm CE}=2.0$",
        "color": "#e85542",
        "med_bold": r"$\boldsymbol{\alpha_{\rm CE}=2.0}$",
        "med_bold2": r"$\boldsymbol{\alpha_{\rm CE}=2.0}$",
        "marker": ">"
    },
    "alpha10": {
        "file": "alpha10",
        "name": r"CE efficiency parameter of $\alpha=10.0$",
        "short": "J",
        "med": r"$\alpha_{\rm CE}=10.0$",
        "color": "#de2d26",
        "med_bold": r"$\boldsymbol{\alpha_{\rm CE}=10.0}$",
        "med_bold2": r"$\boldsymbol{\alpha_{\rm CE}=10.0}$",
        "marker": "^"
    },
    "optimisticCE": {
        "file": "optimistic",
        "name": "HG donor stars initiating CE survive CE",
        "short": "K",
        "med": r"\shortstack{optimistic\\ CE}",  # Fixed line break
        "color": "#bd5d4c",
        "med_bold": r"\shortstack{\textbf{optimistic} \\ \textbf{CE}}",
        "med_bold2": r"\shortstack{\textbf{optimistic} \textbf{CE}}",
        "marker": "8"
    },
    "rapid": {
        "file": "rapid",
        "name": "Fryer rapid SN remnant mass prescription",
        "short": "L",
        "med": r"rapid SN",
        "color": "#016450",
        "med_bold": r"\textbf{rapid SN}",
        "med_bold2": r"\textbf{rapid SN}",
        "marker": "s"
    },
    "maxNSmass2_0": {
        "file": "maxNSmass2_0",
        "name": r"Maximum NS mass = 2.0 ${\rm M_{\odot}}$",
        "short": "M",
        "med": r"\shortstack{max $\text{m}_{\rm NS}$ \\ $2.0 \, {\rm M_{\odot}}$}", 
        "color": "#f6eff7",
        "med_bold": r"\shortstack{\textbf{max} $\boldsymbol{\textbf{m}_{\rm NS}}$ \\ $\boldsymbol{2.0 \, {\rm M_{\odot}}}$}",
        "med_bold2": r"\shortstack{\textbf{max} $\boldsymbol{\textbf{m}_{\rm NS}}$ $\boldsymbol{2.0 \, {\rm M_{\odot}}}$}",
        "marker": "d", 
    },
    "maxNSmass3_0": {
        "file": "maxNSmass3_0",
        "name": r"Maximum NS mass = 3.0 ${\rm M_{\odot}}$",
        "short": "N",
        "med": r"\shortstack{max $\text{m}_{\rm NS}$ \\ $3.0 \, {\rm M_{\odot}}$}", 
        "color": "#d0d1e6",
        "med_bold": r"\shortstack{\textbf{max} $\boldsymbol{\textbf{m}_{\rm NS}}$ \\ $\boldsymbol{3.0 \, {\rm M_{\odot}}}$}",
        "med_bold2": r"\shortstack{\textbf{max} $\boldsymbol{\textbf{m}_{\rm NS}}$ $\boldsymbol{3.0 \, {\rm M_{\odot}}}$}",
        "marker":"D"
    },
    "noPISN": {
        "file": "noPISN",
        "name": "PISN and pulsational-PISN not implemented",
        "short": "O",
        "med": r"no PISN",
        "color": "#a6bddb",
        "med_bold": r"\textbf{no PISN}",
        "med_bold2": r"\textbf{no PISN}",
        "marker":"."
    },
    "ccSNkick_100km_s": {
        "file": "ccSNkick_100km_s",
        "name": r"$\sigma_{\rm RMS}^{\rm 1D} = 100 \ {\rm km\ s^{-1}}$ for core-collapse supernova",
        "short": "P",
        "med": r"$\sigma_{\rm cc}$\\$100 \, {\rm km \, s^{-1}}$",
        "color": "#02818a",
        "med_bold": r"$\boldsymbol{\sigma_{\mathrm{cc}}} \\ \boldsymbol{100 \ \mathrm{km} \ \mathrm{s}^{-1}}$",
        "med_bold2": r"$\boldsymbol{\sigma_{\mathrm{cc}}=100 \ \mathrm{km} \ \mathrm{s}^{-1}}$",
        "marker": "1"
    },
    "ccSNkick_30km_s": {
        "file": "ccSNkick_30km_s",
        "name": r"$\sigma_{\rm RMS}^{\rm 1D} = 30 \ {\rm km\ s^{-1}}$ for core-collapse supernova",
        "short": "Q",
        "med": r"$\sigma_{\rm cc}$\\$30 \, {\rm km \, s^{-1}}$",
        "color": "#3690c0",
        "med_bold": r"$\boldsymbol{\sigma_{\mathrm{cc}}} \\ \boldsymbol{30 \ \mathrm{km} \ \mathrm{s}^{-1}}$",
        "med_bold2": r"$\boldsymbol{\sigma_{\mathrm{cc}}=30 \ \mathrm{km} \ \mathrm{s}^{-1}}$",
        "marker" : "2"
    },
    "noBHkick": {
        "file": "noBHkick",
        "name": "BHs receive no natal kicks",
        "short": "R",
        "med": r"\shortstack{no BH\\ kicks}",  # Fixed line break
        "color": "#67a9cf",
        "med_bold": r"\shortstack{\textbf{no BH} \\ \textbf{kicks}}",
        "med_bold2": r"\shortstack{\textbf{no BH} \textbf{kicks}}",
        "marker": "3"
    },
    "wolf_rayet_multiplier_0_1": {
        "file": "wolf_rayet_multiplier_0_1",
        "name": r"Wolf-Rayet wind factor $f_{\rm WR} = 0.1$",
        "short": "S",
        "med": r"$f_{\rm WR}=0.1$",
        "color": "#756bb1",
        "med_bold": r"$\boldsymbol{f_{\mathrm{WR}}=0.1}$",
        "med_bold2": r"$\boldsymbol{f_{\mathrm{WR}}=0.1}$",
        "marker" : "|"
    },
    "wolf_rayet_multiplier_5": {
        "file": "wolf_rayet_multiplier_5",
        "name": r"Wolf-Rayet wind factor $f_{\rm WR} = 5.0$",
        "short": "T",
        "med": r"$f_{\rm WR}=5.0$",
        "color": "#534c7b",
        "med_bold": r"$\boldsymbol{f_{\mathrm{WR}}=5.0}$",
        "med_bold2": r"$\boldsymbol{f_{\mathrm{WR}}=5.0}$",
        "marker": "_"
    }
}
import numpy as np
import pandas as pd
from bseFlow.file_processing import multiprocess_files
from handy import split_by_DCO_type, record_formation_channels
from totalMassEvolvedPerZ import totalMassEvolvedPerZ

def calculateFormationRates(file, weights=None, DCOtype=None):
    """
    Creates a csv of formation rates and specific formation channel rates
    """
    model = file.split("/")[-2]
    
    _, totalMass = totalMassEvolvedPerZ(pathCOMPASh5=file, Mlower=5., Mupper=150.)
    
    files_and_fields = [
        (file, 'doubleCompactObjects', ['Metallicity1', 'weight'],
            {}),
        ]

    dco_seeds, metallicities, weights = multiprocess_files(files_and_fields, return_df=False, num_processes=2, preview=False)

    channels = record_formation_channels(dco_seeds, file=file, dco_only=True)

    unique_Z = np.unique(metallicities)
    if DCOtype:
        dco_mask = split_by_DCO_type(dco_seeds, mask_type=DCOtype, file=file)
        metallicities = metallicities[dco_mask]
        weights = weights[dco_mask]
        channels = channels[dco_mask]

    formationRateTotal = np.zeros(len(unique_Z))
    formationRateClassic = np.zeros(len(unique_Z))
    formationRateOnlyStableMT = np.zeros(len(unique_Z))
    formationRateSingleCE = np.zeros(len(unique_Z))
    formationRateDoubleCE = np.zeros(len(unique_Z))
    formationRateOther = np.zeros(len(unique_Z))

    for index, Z in enumerate(unique_Z):
        if Z in metallicities:
            maskZ = (metallicities==Z)
            formationRateTotal[index] = np.sum(weights[maskZ])

            filtered_channels = ['0', '1', '2', '3', '4']

            channel_rates = []
            for channel in filtered_channels:
                channel_mask = (metallicities==Z) & (channels == int(channel))
                channel_rates.append(np.sum(weights[channel_mask]))

            formationRateOther[index], formationRateClassic[index], formationRateOnlyStableMT[index], \
            formationRateSingleCE[index], formationRateDoubleCE[index] = channel_rates
        else:
            formationRateTotal[index] 		    = 0
            formationRateClassic[index]         = 0
            formationRateOnlyStableMT[index]    = 0
            formationRateSingleCE[index]        = 0
            formationRateDoubleCE[index]        = 0
            formationRateOther[index]           = 0  

    formationRateTotal = np.divide(formationRateTotal, totalMass)
    formationRateClassic = np.divide(formationRateClassic, totalMass)
    formationRateOnlyStableMT = np.divide(formationRateOnlyStableMT, totalMass)
    formationRateSingleCE = np.divide(formationRateSingleCE, totalMass)
    formationRateDoubleCE = np.divide(formationRateDoubleCE, totalMass)
    formationRateOther = np.divide(formationRateOther, totalMass)

    df = pd.DataFrame(list(zip(unique_Z, formationRateTotal, formationRateClassic, formationRateOnlyStableMT,
                              formationRateSingleCE,  formationRateDoubleCE, formationRateOther)),
              columns=['Z', 'FR_Total','FR_Classic', 'FR_OnlyStableMT', 'FR_SingleCE', 'FR_DoubleCE', 'FR_Other'])

    df.to_csv(f"{model}_formationRates_{DCOtype}.csv")
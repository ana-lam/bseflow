import h5py as h5
import numpy as np
import formation_channels
from calculate_rates import calculate_simulation_rates, clean_buggy_systems
from file_processing import multiprocess_files, create_compact_h5
from utils import in1d

file = '/mnt/home/alam1/ceph/xfer/fiducial/COMPASOutput.h5'
fdata = h5.File(file, 'r')
sys_seeds = fdata['systems']['SEED'][...].squeeze()
print("sys_seeds", sys_seeds)
fdata.close()

cleaned_seeds = clean_buggy_systems(file, selected_seeds=None)
channels, masks = formation_channels.identify_formation_channels(file, selected_seeds=cleaned_seeds)



single_core_seeds = cleaned_seeds[masks['single_core']]
double_core_seeds = cleaned_seeds[masks['double_core']]
print(single_core_seeds.shape, double_core_seeds.shape)

print("now creating h5")

create_compact_h5('/mnt/home/alam1/ceph/xfer/fiducial/COMPASOutput.h5', '/mnt/home/alam1/ceph/COMPASOutput_DoubleCore.h5', double_core_seeds)
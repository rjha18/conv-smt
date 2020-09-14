from data_source import data_source

data_source = data_source(kernel_size=15, sigma=1.5)


input_dir = './Data/bear/'
npy_fnm = './Data/npy/b.npy'
processed_npy_fnm = './Data/bear-processed.npy'


data_source.frames_to_npy(input_dir, npy_fnm, start=1, subsample=1, pre_h=0,
                          suff_h=0, pre_w=0, suff_w=0, frames_threshold=10000)
data_source.process_npy(npy_fnm, processed_npy_fnm)

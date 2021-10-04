#!/usr/bin/env python
    
import h5py
import numpy as np

#dataset_name is intended to be exp_frames or dark_frames
def write_data(file_name, dataset_name, data, index_list, total_indexes):

    print(type(data[0,0,0]))

    print("Writing data into " + str(file_name))

    fname = file_name

    fid = h5py.File(fname, 'a')

    with h5py.File(fname, 'a') as f:

       data_group = "entry_1/data_1/"
       dataset = data_group + dataset_name 
        
       if not dataset in fid: 
           print("Creating data group " + str(dataset))
           fid.create_dataset(dataset, (total_indexes,) + data[0].shape, dtype='uint16') #we create the full dataset size

       #Here we generate a proper index pull map, with respect to the input
       print(dataset_name)
       print(data[0].shape)
       print(data.shape)
       print(index_list)
       m = min(index_list) #we need this to reshuffle the indexes and to index properly on the output file
       print(m)
       index_list = np.array([ np.where(index_list==i)[0][0] for i in range(m, m + len(index_list))])
       print(index_list)
       fid[dataset][m:m + data.shape[0]] = data[index_list]

def write_metadata(file_name, metadata):

    print("Writing metadata into " + str(file_name))
    fname = file_name

    fid = h5py.File(fname, 'a')

    with h5py.File(fname, 'a') as f:

       group = "metadata/"
       f.create_dataset(group, data = metadata)



#!/usr/bin/env python
    
import h5py

#dataset_name is intended to be exp_frames or dark_frames
def write_data(file_name, dataset_name, data, index_list, total_indexes):

    fid = h5py.File(file_name, 'a')

    with h5py.File(file_name, 'a') as f:

       data_group = "entry_1/data_1/"
       dataset = data_group + dataset_name 
        
       if not data_group in fid: 
           fid.create_group(data_group)
           fid.create_dataset(dataset, (total_indexes,) + data.shape[1:], dtype='float32') #we create the full dataset size

       #Here we generate a proper index pull map, with respect to the input
       print(dataset_name)
       print(index_list)
       index_list = npo.array([ npo.where(index_list==i)[0][0] for i in range(0,len(index_list))])
       print(index_list)
       fid[dataset][:,:,:] = data[index_list,:,:]





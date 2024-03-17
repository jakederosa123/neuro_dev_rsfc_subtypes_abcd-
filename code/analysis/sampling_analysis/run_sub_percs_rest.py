#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
#sys.path.append('/pl/active/banich/studies/Relevantstudies/abcd/env/lib/python3.7/site-packages')
sys.path.append('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/')
from functions import *

import random

# Set the seed value for random number generator
random.seed(42)


# In[ ]:


datatype = 'rest'
which = 'include'

# In[ ]:


#full_sample = pd.read_csv(f'/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/subtype_data/baseline/fullsample_{datatype}_baseline_07282023_{which}.csv').iloc[:, 1:]
#sample1 = pd.read_csv(f'/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/subtype_data/baseline/sample1_{datatype}_baseline_07282023_{which}.csv').iloc[:, 1:]
#sample2 = pd.read_csv(f'/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/subtype_data/baseline/sample2_{datatype}_baseline_07282023_{which}.csv').iloc[:, 1:]


# In[ ]:


def sampling(data, perc): #1 
    '''ramdonly select sample data with replacement for a single bootstrap'''
    import random
    seed = random.randint(0, 1000000)
    np.random.seed(seed)
    total_n=data.shape[0] # total number of subjects 
    this_index = np.random.choice(total_n, int(total_n * perc), replace=False) # randomly select the subjects' index with replacement for this bootstrapt 
    sampled_data = data.iloc[this_index] # use the selected index to slice the data for this bootstrapt
    return sampled_data   


# In[ ]:



import os 

outfolder = f'/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/{datatype}_{which}/'
os.system(f'mkdir -p {outfolder}/')


run_sampling_code = """
sample = ['full_sample', 'sample1', 'sample2']

lists = []
save_paths = []
for s,d in zip(sample, [full_sample, sample1, sample2]):
    for n in range(100):
        for i in np.linspace(0,1,11)[1:10].round(1):
            os.system(f'mkdir -p {outfolder}/{i}')
            sampling(d,i).to_csv(outfolder+'/'+str(i)+'/'+s+'_'+str(i)+'_'+str(n)+'.csv')           

"""
# In[ ]:

#run_sampling_code = """
from glob import glob
paths = f'/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/{datatype}_{which}/*/*/Output/Results/*_Full_Subtypes.csv'
paths_sorted = sorted(glob(paths, recursive = True))
#list(filter(lambda k: '0.3' in k, paths_sorted))
path_num = len(paths_sorted)-1

import sys
import ruamel.yaml

yaml = ruamel.yaml.YAML()
# yaml.preserve_quotes = True
with open('/pl/active/banich/studies/Relevantstudies/abcd/data/BagPype/scripts/bagpype_template.yaml') as fp:
    data = yaml.load(fp)

from glob import glob
paths = f'/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/{datatype}_{which}/*/*.csv'
paths_sorted = sorted(glob(paths, recursive = True))

for i in paths_sorted[path_num:]:
    
    if '/sample' in i:
        data['data_path'] = i.split('/sample')[0] #+ '/sample'
    elif '/full_sample' in i:
        data['data_path'] = i.split('/full_sample')[0] #+ '/full_sample'
    #else:
    #    data['data_path'] = i
    
    #data['data_path'] = i.split('/sample')[0]

    data['batch'] = (i.split(f'/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/sampling/{datatype}_{which}/')[1]
     .split('/')[1]
     .split('.csv')[0]) 
    
    from pathlib import Path
    output = Path('/pl/active/banich/studies/Relevantstudies/abcd/data/BagPype/scripts/bagpype_template.yaml')
    yaml.dump(data, output)
    
    print(data['data_path'])
    import subprocess
    pcall = 'sh /pl/active/banich/studies/Relevantstudies/abcd/data/BagPype/scripts/bagit.sh'
    subprocess.check_call(pcall, shell=True)





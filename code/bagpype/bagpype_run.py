#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from bagpype_call import *

outfolder = (); path = ()


'''
for i in params['batch']:
    batch_path = params['data_path']
    batch_name = f'{i}'
    outfolder= batch_path+'/'+batch_name
    path = f'{outfolder}/{i}.csv' 
    os.system(f'mkdir -p {batch_path}/{i}')
    
    for d in [batch_path+"/"+batch_name +'.csv']:
        shutil.copy(d, outfolder)
          
        bagpype(outfolder = outfolder,
                batch = i, 
                path = path, 
                n_straps = params['n_straps'], 
                ID = params['ID'], 
                Boot = params['Boot'], 
                algorithm = params['algorithm'], 
                distance_metric = params['distance_metric'], 
                k = params['k'],
                resolution_parameter=params['resolution_parameter']
                ) 
'''

if params['Boot'] == True:
    params['Boot'] = 'Yes'


if params['Boot'] == False:
    params['Boot'] = 'No'
    

output = []
#changed specifically for the sampling! Unhash once sampling is done
for i,j in zip(params['batch'], params['resolution_parameter']):
    output.append([i, str(j)])
00
#for i,j in zip(params['batch'], params['resolution_parameter']):
#    output.append([params['batch'], str(j)])
    
of = pd.DataFrame(output)
of.columns = ['df', 'limit']

for i in range(0,of.shape[0]):
    batch_path = params['data_path']
    new_name = str(of.loc[i]['df'])
    limit = float(of.loc[i]['limit'])
    batch_name = new_name
    outfolder= batch_path+'/'+batch_name
    path = f'{outfolder}/{new_name}.csv' 
    os.system(f'mkdir -p {batch_path}/{new_name}')
    
    for d in [batch_path+"/"+batch_name +'.csv']:
        shutil.copy(d, outfolder)
          

        bagpype(outfolder = outfolder,
                batch = new_name, 
                path = path, 
                n_straps = params['n_straps'], 
                ID = params['ID'], 
                Boot = params['Boot'], 
                algorithm = params['algorithm'], 
                distance_metric = params['distance_metric'], 
                k = params['k'],
                resolution_parameter=limit
                ) 

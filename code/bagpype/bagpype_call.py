import sys
sys.path.append('/pl/active/banich/studies/Relevantstudies/abcd/bagpype/lib/python3.7/site-packages')

import os
import sys
import glob
import pandas as pd
import sklearn
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import train_test_split
from time import time
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
#import seaborn as sns
from igraph import *
import subprocess
from sklearn import datasets
from sklearn.metrics import *
import shutil
import yaml
params = yaml.safe_load(open('bagpype_template.yaml'))
from importlib import resources
import io 
from scipy.spatial import distance
from sklearn.neighbors import DistanceMetric
import math
from functools import reduce
from sklearn.preprocessing import MinMaxScaler
# Perform min-max scaling

                                        
                                        
#=======================================================================================================#
def reduce_memory_usage(df, verbose=False):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df

#=======================================================================================================#
def uni(list1):
    """
    Function to get unique values of list
    Returns all unique items
    """
    unique_list = [] # initialize a null list

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)

    return(unique_list)

#=======================================================================================================#
def hadamard(nearest):
    """
    This function converts a directed adjacency matrix to a matrix of Hadamard distances
    based on the overlap of neighbors. The implementation is efficiently based
    on matrix multiplications.
    A an N-by-N adjacency matrix holding TRUE or 1 values for edges an N-by-N matrix with
    the Hadamard coefficient of neighbor overlap.
    """

    common = nearest.T @ nearest
    ranks = np.outer(np.diag(common), np.ones(nearest.shape[0]))
    neighborUnion = ranks + ranks.T - common
    G = common / neighborUnion
    np.fill_diagonal(G, 0)
    
    return G

#=======================================================================================================#
def topMax(x, N):
    """
    find Nth largest number in an array
    """
    
    L = len(x)
    assert N < L, 'Number of neighbors cannot be larger than length of data'
    
    while L != 1:
        initial_guess = x[0]
        top_list = x[x > initial_guess]
        bottom_list = x[x < initial_guess]
        
        topL = len(top_list)
        bottomL = len(bottom_list)
        
        if (topL < N) and (L - bottomL >= N):
            x = initial_guess
            break
        
        if topL >= N:
            x = top_list
        else:
            x = bottom_list
            N = N - L + bottomL
        L = len(x)
    
    return x

#=======================================================================================================#
def bottomMin(x, N):
    """
    find Nth smallest number in an array
    """

    return np.round(topMax(x, len(x) - N + 1))

#=======================================================================================================#
def find_neighbors(D, k):
    """
    Tranform distance matrix to binary k-nearest neighbors graph
    """
    
    nearest = np.zeros(D.shape)
    for i in range(nearest.shape[1]):
        nearest[:,i] = np.round(D[:,i]) <= bottomMin(np.round(D[:,i]), k)
   
    
    return nearest

#=======================================================================================================#        
def pheno_clust(filepath=None, subset=None, X=None, outfile=None,
                verbose=True, algorithm=None, distance=None, k=None, resolution_parameter=None):
    """
    PhenoClust pipeline:                                                                     
    This method clusters sample data using a two-step approach. In the first step it converts the high dimensional data into a knn-nearest neighbor graph, 
    thus connecting samples that have a similar profile. The edges in knn-nearest neighbors graph are given weights according also known as the Hadamard
    index. In the second step the clustering algorithm of choice is applied to this graph in order to find samples that share similar spaces in multi-dimensional space.
    """
    assert filepath or X is not None, 'Either a matrix or a filepath to the data must be passed in'
    
    if verbose:
        tic = time()
    
    if filepath:
        assert subset, 'If filepath to dataframe is passed, you must also include the subset of columns that you intend to pass into the algorithm'
        
        tp = pd.read_csv(filepath, iterator=True, chunksize=100000)  # gives TextFileReader
        df = pd.concat(tp, ignore_index=True)
        df = df[subset]
        df.dropna(inplace=True)
        X = np.array(df)
        
    #### Standardize data
    X_Z = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, dtype=np.float64, ddof=1, keepdims=True)
    
    #### Pick distance metric for 
    if distance == "spearman":  
        
        #X = pd.DataFrame(X)
        #for col in X.columns:
        #    X[col] = (X[col] - X[col].mean())/X[col].std(ddof=0)
        
        #scaler = MinMaxScaler(feature_range=(0,1))
        #X = scaler.fit_transform(X)
                     
        D, rho = spearmanr(X_Z, axis=1)
        D = np.round((X_Z.shape[1]**3 - X_Z.shape[1]) * (1 - D) / 6).astype(np.int)
        
    if distance == "euclidean":
        X = pd.DataFrame(X)
        for col in X.columns:
            X[col] = (X[col] - X[col].mean())/X[col].std(ddof=0)
    
        D = euclidean_distances(np.array(X))
    
    if distance == "mahalanobis":
        X = pd.DataFrame(X)
        for col in X.columns:
            X[col] = (X[col] - X[col].mean())/X[col].std(ddof=0)
            
        D = DistanceMetric.get_metric('mahalanobis', V=np.cov(np.array(X).T)).pairwise(np.array(X))
    
    b = np.ceil(np.log2(X_Z.shape[0]) + 1)
    
    if k is None:
        k = np.ceil(X_Z.shape[0] / b).astype(np.int)
        
    if k == "sqrt":
            k = np.ceil(math.sqrt(D.shape[0]))
    else: 
        k = k 
        
    nearest = find_neighbors(D, k + 1)
    G = hadamard(nearest)  
    gframe = pd.DataFrame(G)
    #gframe = reduce_memory_usage(gframe)
    graph = Graph.Weighted_Adjacency(gframe.values.tolist(), mode=ADJ_UNDIRECTED, attr="weight")
   
    #### Pick community detection method
    
    if algorithm == "infomap":
        alg = graph.community_infomap()
        Q = 0
        
    if algorithm == "leiden":
        if resolution_parameter is None:
            alg = graph.community_leiden(objective_function = "modularity", weights=graph.es['weight'])
        else: 
            alg = graph.community_leiden(objective_function = "modularity", weights=graph.es['weight'], resolution_parameter=resolution_parameter)
        
        Q = graph.modularity(alg, weights=graph.es['weight']) 
     
    if algorithm == "louvain":
        if resolution_parameter is None:
            alg = graph.community_multilevel(weights=graph.es['weight'])
        else: 
            alg = graph.community_multilevel(weights=graph.es['weight'], resolution_parameter=resolution_parameter)
    
        Q = graph.modularity(alg, weights=graph.es['weight']) 

    print("Found {} unique clusters ".format(len(uni(alg.membership)))+"------ Q = "+str(round(Q,3))+"")
    C = np.array(alg.membership).astype(int)
    
    if verbose:
        toc = time()
        print('BagPype completed in: %.2f seconds' % (toc - tic))
    return C, Q


    
#=======================================================================================================#

def adjacency_matrix(cluster_pred):
    """
    function from PyBASC package https://github.com/AkiNikolaidis/PyBASC
    """
    from scipy import sparse
    
    x = cluster_pred.copy()
    if(len(x.shape) == 1):
        x = x[:, np.newaxis]
        
    # Force the cluster indexing to be positive integers
    if x.min() <= 0:
        x += -x.min() + 1
        
    A = np.dot(x**-1., x.T) == 1
    A = sparse.csr_matrix(A, dtype=bool)
        
    return A
    
#=======================================================================================================#
def bagging_adjacency_matrixes(csv_folder, split, data_frame_path, out_dir, out_dir_stab, 
                               subset, resolution_parameter, algorithm):
    '''csv_folder: folder that contains all the csv output from python louvain.
       split: string in the csv files' names that specifies the split eg.Split_1'''
    
    import pandas as pd
    from functools import reduce
    
    def create_adj(file):
        # New Bagging Code ------------------------------------------------------------------------------------
        l_adm_all=[] 
        for f in file:
            df = pd.read_csv(f).iloc[:, 1:].rename(columns = {"0":'Key', "1":'cluster'}).drop_duplicates('Key')
            df['cluster'] = df['cluster'].astype(int)
            df['Key'] = df['Key'].astype(int)
         
            labels_A=adjacency_matrix(np.array(df['cluster'])).astype('int').todense()
           
            np.fill_diagonal(labels_A, 0)

            labels_A = pd.DataFrame(labels_A)
            labels_A.columns = list(df['Key'])
            labels_A = labels_A.T
            labels_A.columns = list(df['Key'])

            l_adm_all.append(labels_A)

            print(f'finished -- {f}')
           
        print('starting adding')
        adj_full = reduce(lambda a, b: a.add(b, fill_value=0), l_adm_all).fillna(0)#.astype(int)
        print('DONE!') 
        #adj_full.to_csv('/pl/active/banich/studies/Relevantstudies/abcd/data/BagPype/scripts/test_mat.csv')
        return adj_full
    
    def create_mask(file):
     
        is_adm_all=[]
       
        for f in file:
            df = pd.read_csv(f).iloc[:, 1:].rename(columns = {"0":'Key', "1":'cluster'}).drop_duplicates('Key')
            df['cluster'] = df['cluster'].astype(int)
            df['Key'] = df['Key'].astype(int)
         
            in_sample_A=adjacency_matrix(np.array(df['Key'])).astype('int').todense()
  
            np.fill_diagonal(in_sample_A, 0)

            in_sample_A = pd.DataFrame(in_sample_A)
            in_sample_A.columns = list(df['Key'])
            in_sample_A = in_sample_A.T
            in_sample_A.columns = list(df['Key'])
            in_sample_A = in_sample_A.replace(0, 1)

            is_adm_all.append(in_sample_A)

            print(f'finished -- {f}')
            
        print('starting adding')
     
        mask_full = reduce(lambda a, b: a.add(b, fill_value=0), is_adm_all).fillna(0)#.astype(int)
        print('DONE!') 
   
        return mask_full
    
    #stab_full = adj_full/mask_full                    
    # End New Bagging Code ---------------------------------------------------------------------------------
    import os
    import pandas as pd
    
    files = sorted(glob.glob(os.path.join(csv_folder, "*.csv")))

    def split_list_into_chunks(file_list, chunk_size):
        return [file_list[i:i + chunk_size] for i in range(0, len(file_list), chunk_size)]

    chunk_size = 20
    # Split the list into chunks
    chunks = split_list_into_chunks(files, chunk_size)
   
    # Ensure the directories exist
    os.makedirs(f'{csv_folder}/Output/Results/adj_matrices', exist_ok=True)
    os.makedirs(f'{csv_folder}/Output/Results/mask_matrices', exist_ok=True)
    
    def create_adj_mask(chunks, csv_folder):
        # Create and save adjacency and mask matrices to CSV files
        for i, file in enumerate(chunks):
            adj_matrix = create_adj(file)
            mask_matrix = create_mask(file)

            adj_matrix.to_csv(f'{csv_folder}/Output/Results/adj_matrices/adj_matrix_{i}.csv')
            mask_matrix.to_csv(f'{csv_folder}/Output/Results/mask_matrices/mask_matrix_{i}.csv')
    
    create_adj_mask(chunks, csv_folder)

    # Get a list of all the CSV files for adjacency and mask matrices
    #del adj_matrix, mask_matrix
    # Get a list of all files in the directory
    file_list = os.listdir(f'{csv_folder}Output/Results/adj_matrices/')
    # Filter out only the CSV files
    csv_files = [os.path.join(csv_folder, 'Output', 'Results', 'adj_matrices', file) for file in file_list if file.endswith('.csv')]
    # Read all CSV files into a list of pandas DataFrames
    adj_matrices = [pd.read_csv(file, index_col=0) for file in csv_files]
    adj_full = reduce(lambda a, b: a.add(b, fill_value=0), adj_matrices).astype(int)
    del adj_matrices

    file_list = os.listdir(f'{csv_folder}Output/Results/mask_matrices/')
    csv_files = [os.path.join(csv_folder, 'Output', 'Results', 'mask_matrices', file) for file in file_list if file.endswith('.csv')]
    mask_matrices = [pd.read_csv(file, index_col=0) for file in csv_files]
    mask_full = reduce(lambda a, b: a.add(b, fill_value=0), mask_matrices).astype(int)
    del mask_matrices

    stab_full = adj_full/mask_full 
    
    old_bagging_code = """"my_data = []
    for f in glob.glob(os.path.join(csv_folder, "*.csv")):
        my_data.append(pd.read_csv(f))
    for i in range(len(my_data)): # loop over my_data
        my_data[i] = my_data[i].rename(columns = {"0":'Key', "1":'cluster'})
        boot=my_data[i].drop_duplicates() #remove duplicates
        df_merge = boot.merge(boot, on='cluster') # create adjacency matrix step1 
        this_adj =pd.crosstab(df_merge.Key_x, df_merge.Key_y) # create adjacency matrix step2
        np.fill_diagonal(this_adj.values, 0) #set the diagonal of the adjacency matrix to 0
        if i ==0:
            adj_full=this_adj #if this is the first adjacency matrix set it as the initial adj_full 
        else:
            adj_full=adj_full.add(this_adj,fill_value=0) # add this adjacency matrix to the full adjacency matrix
        boot_mask=my_data[i].drop_duplicates()
        boot_mask.cluster=boot_mask.cluster.replace(boot_mask.cluster.unique(),np.ones(boot_mask.cluster.unique().shape))#replace all clusters with 1
        df_merge2 = boot_mask.merge(boot_mask, on='cluster')
        mask_adj =pd.crosstab(df_merge2.Key_x, df_merge2.Key_y)
        np.fill_diagonal(mask_adj.values, 0) #set the diagonal of the adjacency matrix to 0
        if i ==0:
            mask_full= mask_adj  #if this is the first adjacency matrix set it as the initial mask_full 
        else:
            mask_full=mask_full.add( mask_adj,fill_value=0) # add this mask adjacency matrix to the full mask adjacency matrix
    stab_full = adj_full.div(mask_full)
    """
    np.fill_diagonal(stab_full.values, 0)
    #Create df Id from matrix column names
    columnsNamesArr = pd.DataFrame(stab_full.columns.values)
    #Prep Graph
    graph = Graph.Weighted_Adjacency(stab_full.values.tolist(), mode=ADJ_UNDIRECTED, attr="weight")
    
    #### Pick community detection method
    if algorithm == "leiden":
        if resolution_parameter is None:
            alg = graph.community_leiden(objective_function = "modularity", weights=graph.es['weight'])
        else: 
            alg = graph.community_leiden(objective_function = "modularity", weights=graph.es['weight'], resolution_parameter=resolution_parameter)
     
    if algorithm == "louvain":
        if resolution_parameter is None:
            alg = graph.community_multilevel(weights=graph.es['weight'])
        else: 
            alg = graph.community_multilevel(weights=graph.es['weight'], resolution_parameter=resolution_parameter)

    Q = graph.modularity(alg, weights=graph.es['weight'])
    print("Found {} unique clusters ".format(len(uni(alg.membership)))+"------ Bagged Q = "+str(round(Q,3))+"")
    Subtypes = pd.DataFrame(alg.membership)
    Subtypes = Subtypes + 1
    Subtypes.loc[:, 'Q'] = Q
    subs = pd.concat([columnsNamesArr.reset_index(drop=True), Subtypes], axis=1)
    subs.columns = ['Key', 'Subtype', 'Q']
    #df = pd.read_csv(data_frame_path)
    tp = pd.read_csv(data_frame_path, iterator=True, chunksize=100000) # gives TextFileReader
    df = pd.concat(tp, ignore_index=True)
    df['Unnamed: 0'] = range(1, df.shape[0] + 1)
    df = df.rename(columns={'Unnamed: 0': 'Key'})
    # create Split 1 
    subs['Key'] = subs['Key'].astype('object')
    df['Key'] = subs['Key'].astype('object')
    
    Split_1 = pd.merge(subs, df, on='Key')
    Split_1.to_csv(out_dir)
    stab_full.to_csv(out_dir_stab)
    return stab_full, Split_1, Q

#=======================================================================================================#
def bagpype(outfolder, batch, path, n_straps, ID, algorithm=None, Boot=None, distance_metric=None, 
            k=None, resolution_parameter=None):
    # set up 
    batch = batch
    resolution_parameter=resolution_parameter
    
    tp = pd.read_csv(path, iterator=True, chunksize=100000)  # gives TextFileReader
    df1 = pd.concat(tp, ignore_index=True)
    df1.dropna(inplace=True)
    #df1 = reduce_memory_usage(df1)
    #df1 = pd.read_csv(path)
    #df1['Key'] = list(range(len(df1))) 
    if Boot == 'No': 
        os.system(f'mkdir -p {outfolder}/Output/')
        os.system(f'mkdir -p {outfolder}/Output/Results/NonBoot')
        #df1 = df1.drop(['Unnamed: 0'], axis=1)
        subset = df1.columns[df1.columns != ID]
        y = np.array(df1[ID],dtype='str')
        
        if resolution_parameter is None:
            communities, Q = pheno_clust(X=np.array(df1[subset]).astype(np.float64), verbose=False, 
                                         algorithm=algorithm, distance = distance_metric, k = k)
        else: 
            communities, Q = pheno_clust(X=np.array(df1[subset]).astype(np.float64), verbose=False, 
                                         algorithm=algorithm, distance = distance_metric, k = k, resolution_parameter=resolution_parameter)
            
        df2 = {ID:y,'Subtype':communities}
        df2 =pd.DataFrame(df2)
        df2.loc[:, 'Q'] = Q
        df2 = pd.merge(df2, df1, on = ID)
        df2['Subtype'] = df2['Subtype'] + 1
        df2.to_csv(f'{outfolder}/Output/Results/NonBoot/NonBoot_{batch}_Full_Subtypes.csv')
    
    if Boot == 'Both':
        df1 = df1.drop([ID], axis=1)
        df1['Unnamed: 0'] = range(1, df1.shape[0] + 1)
        df1 = df1.rename(columns={'Unnamed: 0': 'Key'})
        subset = df1.columns[df1.columns != 'Key']
        df = df1[subset]
        y = np.array(df1['Key'],dtype='str')
        n_straps = n_straps
        n = df.shape[0]
        b_idx = np.zeros((n_straps, n))
        # bootstrapping
        for i in range(n_straps):
            random_state = np.random.RandomState()
            b_idx[i] = random_state.randint(0, high= n, size=n)
        b_idx = b_idx.astype(np.int)
        y_boot = np.zeros(b_idx.shape)
        for i in range(b_idx.shape[0]):
            y_boot[i] = y[b_idx[i]]
        #create directories
        os.system(f'mkdir -p {outfolder}/Data/')
        os.system(f'mkdir -p {outfolder}/Output/')
        os.system(f'mkdir -p {outfolder}/Output/Results/')
        os.system(f'mkdir -p {outfolder}/Output/Results/Boot')
        os.system(f'mkdir -p {outfolder}/Output/Results/Boot_Q')
        #save data 
        df.to_csv(f'{outfolder}/Data/temp.csv')
        np.save(f'{outfolder}/Output/Results/y_boot.npy',y_boot)
        np.save(f'{outfolder}/Output/Results/b_idx.npy',b_idx)
        limit_frame_i = pd.DataFrame([resolution_parameter]*10)
        limit_frame_i.columns = ['limit']
        limit_frame_i.to_csv(f'{outfolder}/Output/Results/res_param.csv')
        
       
        temp_txtfile = np.arange(0,n_straps,1).astype(str)
        np.savetxt(f'{outfolder}/Output/Results/temp_txtfile.txt', temp_txtfile,fmt='%s')
        #parallel call 
        tstart = time()
        
        pcall = 'bash ' + params['pype_path'] + '/BagPype/scripts/bagpype_bash_parallel.sh %s' % batch
        #pcall = 'bash ' + params['pype_path'] + '/BagPype/scripts/bagpype_bash_parallel.sh %s %s' % (batch, resolution_parameter)
        subprocess.check_call(pcall, shell=True)
        tend = time()
        print('Time to run %s bootstraps: ' % n_straps, tend-tstart)
        #Bagging 
        csv_folder = f'{outfolder}/Output/Results/Boot/'
        split = batch
        data_frame_path = path
        out_dir = f'{outfolder}/Output/Results/{batch}_Full_Subtypes.csv'
        out_dir_stab = f'{outfolder}/Output/Results/{batch}_Full_Stab.csv'
        stab_full, Split_1, Q = bagging_adjacency_matrixes(csv_folder, split, data_frame_path, out_dir, out_dir_stab, subset, resolution_parameter, algorithm)
        
        os.system(f'mkdir -p {outfolder}/Output/Results/NonBoot')
        df1 = pd.read_csv(path)
        df1 = df1.drop(['Unnamed: 0'], axis=1)
        subset = df1.columns[df1.columns != ID]
        y = np.array(df1[ID],dtype='str')
        if resolution_parameter is None:
            communities, Q = pheno_clust(X=np.array(df1[subset]).astype(np.float64), verbose=False, algorithm = algorithm, distance = distance_metric, k = k)
        else: 
            communities, Q = pheno_clust(X=np.array(df1[subset]).astype(np.float64), verbose=False, algorithm = algorithm, distance = distance_metric, k = k, resolution_parameter=resolution_parameter)

        df2 = {ID:y,'Subtype':communities}
        df2 =pd.DataFrame(df2)
        df2.loc[:, 'Q'] = Q

        df2 = pd.merge(df2, df1, on = ID)
        df2['Subtype'] = df2['Subtype'] + 1
        df2.to_csv(f'{outfolder}/Output/Results/NonBoot/NonBoot_{batch}_Full_Subtypes.csv')
    
    if Boot == 'Yes':
    
        df1 = df1.drop([ID], axis=1)
        df1['Unnamed: 0'] = range(1, df1.shape[0] + 1)
        df1 = df1.rename(columns={'Unnamed: 0': 'Key'})
        subset = df1.columns[df1.columns != 'Key']
        df = df1[subset]
        y = np.array(df1['Key'],dtype='str')
        n_straps = n_straps
        n = df.shape[0]
        b_idx = np.zeros((n_straps, n))
        # bootstrapping
        for i in range(n_straps):
            random_state = np.random.RandomState()
            b_idx[i] = random_state.randint(0, high= n, size=n)
        b_idx = b_idx.astype(np.int)
        y_boot = np.zeros(b_idx.shape)
        for i in range(b_idx.shape[0]):
            y_boot[i] = y[b_idx[i]]
        #create directories
        os.system(f'mkdir -p {outfolder}/Data/')
        os.system(f'mkdir -p {outfolder}/Output/')
        os.system(f'mkdir -p  {outfolder}/Output/Results/')
        os.system(f'mkdir -p  {outfolder}/Output/Results/Boot')
        os.system(f'mkdir -p  {outfolder}/Output/Results/Boot_Q')

        #save data 
        df.to_csv(f'{outfolder}/Data/temp.csv')
        np.save(f'{outfolder}/Output/Results/y_boot.npy',y_boot)
        np.save(f'{outfolder}/Output/Results/b_idx.npy',b_idx)
        limit_frame_i = pd.DataFrame([resolution_parameter]*10)
        limit_frame_i.columns = ['limit']
        limit_frame_i.to_csv(f'{outfolder}/Output/Results/res_param.csv')
        
        temp_txtfile = np.arange(0,n_straps,1).astype(str)
        np.savetxt(f'{outfolder}/Output/Results/temp_txtfile.txt', temp_txtfile,fmt='%s')
        #parallel call 
        
        tstart = time()
        pcall = 'bash ' + params['pype_path'] + '/BagPype/scripts/bagpype_bash_parallel.sh %s' % batch
        subprocess.check_call(pcall, shell=True)
        tend = time()
        print('Time to run %s bootstraps: ' % n_straps, tend-tstart)
        
        #Bagging 
        csv_folder = f'{outfolder}/Output/Results/Boot/'
        split = batch
        data_frame_path = path
        out_dir = f'{outfolder}/Output/Results/{batch}_Full_Subtypes.csv'
        out_dir_stab = f'{outfolder}/Output/Results/{batch}_Full_Stab.csv'
        stab_full, Split_1, Q = bagging_adjacency_matrixes(csv_folder, split, data_frame_path, out_dir, out_dir_stab, subset, resolution_parameter, algorithm)
        
     
    if Boot == 'bagging':
        
        df1 = df1.drop([ID], axis=1)
        df1['Unnamed: 0'] = range(1, df1.shape[0] + 1)
        df1 = df1.rename(columns={'Unnamed: 0': 'Key'})
        subset = df1.columns[df1.columns != 'Key']
        df = df1[subset]
        
        csv_folder = f'{outfolder}/Output/Results/Boot/'
        split = batch
        data_frame_path = path
        out_dir = f'{outfolder}/Output/Results/{batch}_Full_Subtypes.csv'
        out_dir_stab = f'{outfolder}/Output/Results/{batch}_Full_Stab.csv'
        stab_full, Split_1, Q = bagging_adjacency_matrixes(csv_folder, split, data_frame_path, out_dir, out_dir_stab, subset, resolution_parameter, algorithm)
         
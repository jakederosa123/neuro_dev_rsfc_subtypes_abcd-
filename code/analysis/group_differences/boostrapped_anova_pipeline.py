#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('/pl/active/banich/studies/Relevantstudies/abcd/env/lib/python3.7/site-packages')
sys.path.append('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/')
from import_data import *
from import_subtypes import *

beh_remove = [
    #'withdrawn_depressed_r', 'somatic_complaints_r', 'anxious_depressed_r', 'rule_breaking_r', 
    #'agressive_r', 'attention_problems_r', 'thought_problems_r', 'social_problems_r', 
    'int_factor', 'ext_factor', 'p_factor'
    #'total_r', 'internalizing_r', 'externalizing_r'
             ]

#beh_remove = ['int_factor', 'ext_factor', 'p_factor']

cog_remove = ['CardSort_r', 'Flanker_r', 'nb_r', 'List_r', 'sst_r', 'nb_r', 'List_r', 'matrix_r', 'PicVocab_r', 'Reading_r', 'Picture_r', 'Pattern_r']

beh_all_names = list(filter(lambda item: item not in beh_remove, beh_all_names))
cog_all_names = list(filter(lambda item: item not in cog_remove, cog_all_names))

def boot_pipeline(df1_idsub, df2_idsub, full_sample_idsub, data_type, n_boots):
    
    def getF(data, var, formula):
        #data[var] = data[var]#.astype('float')
        import re
        data = data[[var] + re.findall(r'\((.*?)\)', formula)].dropna()
        model = ols(var + formula, data=data).fit()
        anova_table = np.array(sm.stats.anova_lm(model, type=2)[['F', 'PR(>F)']])[0]
        return anova_table


    def getposthoc(data, var, formula):
        import re
        data = data[[var] + re.findall(r'\((.*?)\)', formula)].dropna()
        #data[var] = data[var]#.astype('float')
        model = ols(var + formula, data=data).fit()
        post_hoc = sp.posthoc_ttest(data, val_col=var, group_col='Subtype',
                                    p_adjust='fdr_bh').sort_index().sort_index(axis = 1)
        ph = np.array(post_hoc)
        tril = np.triu_indices(len(ph))
        ph[tril] = np.nan    
        post_hocsm = pd.DataFrame(ph).melt().dropna().reset_index(drop=True)

        new_list=[]
        y = list(itertools.combinations(list(post_hoc.columns),2))
        for i in range(len(list(itertools.combinations(list(post_hoc.columns),2)))):
            combo = str(y[i][0])+ "-"+str(y[i][1])
            new_list.append(combo)

        post_hocsm['variable'] = new_list
        #post_hocsm['variable'] = list(itertools.combinations(list(post_hoc.columns),2))
        post_hocsm.columns = ['Group', 'pvalue']
        post_hocsm = post_hocsm.query('pvalue < .05')


        final_sig = post_hocsm[['Group']].T
        final_sig['Sig_Post'] = final_sig[final_sig.columns[0:]].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)
        final_sig['var'] = var

        return final_sig[['var', 'Sig_Post']].reset_index(drop=True)


    def run_anova(data, test_vars, formula):

        if type(test_vars) == str:
            f_tests_df = pd.DataFrame(getF(data, test_vars, formula)).T
        else:
            f_tests=[]
            for i in test_vars:
                f_tests.append(getF(data, i, formula))
            f_tests_df = pd.DataFrame(f_tests)

        f_tests_df['var'] = test_vars
        f_tests_df.columns = ['fval', 'pval', 'var']
        f_tests_df = f_tests_df[['var', 'fval', 'pval']]

        sig_f_tests_df = f_tests_df[f_tests_df.apply(lambda x: x['pval'] < .05, axis=1)]

        if sig_f_tests_df.shape[0] != 0:
            if type(test_vars) == str:
                sig_post_hocs = getposthoc(data, test_vars, formula)
            else:
                post_hocs=[]
                for i in list(sig_f_tests_df['var']):
                    post_hocs.append(getposthoc(data, i, formula))
                sig_post_hocs = pd.concat(post_hocs)

            final_sig_frame = pd.merge(sig_f_tests_df, sig_post_hocs, on ='var')

            final_sig_list = list(sig_post_hocs['var'])

            not_sig_tests = f_tests_df.query("var not in @final_sig_list")
            not_sig_tests['Sig_Post'] = ''

            final_tests_df = pd.concat([final_sig_frame, not_sig_tests]).reset_index(drop=True)#.round(3)

        else:
            final_tests_df = f_tests_df
            final_tests_df['Sig_Post'] = ''

        return final_tests_df

    def format_sample_anovas(data): 

        xx = (pd.concat(data)
          .query('Sample != "Full_Sample"')
          .replace('', np.NaN)
          .sort_values('var')
         )

        robust_vars = []
        for i in list(xx['var'].unique()):
            tt = (xx.query('Sample != "Full_Sample"')[['var', 'Sig_Post', 'Sample']]
              .query('var == "' + i + '"')) 

            tt_sample1 = tt.iloc[0][['Sig_Post']].str.split(',', expand=True)
            tt_sample2 = tt.iloc[1][['Sig_Post']].str.split(',', expand=True)

            list(tt_sample1.melt()['value'])
            list(tt_sample2.melt()['value'])

            if len([x for x in list(tt_sample1.melt()['value']) if x in list(tt_sample2.melt()['value'])]) > 0:
                robust = True
            else:
                robust = False

            robust_vars.append(robust)

        robust_df = pd.DataFrame(np.array([list(xx['var'].unique()), robust_vars])).T
        robust_df.columns = ['var', 'Robust']
        final_output = pd.merge(xx, robust_df, on = 'var').replace(np.NaN, '')
        final_output['Robust'] = case_when((final_output[['Robust']] == 'False'), '', default=final_output[['Robust']])

        final_output_full_sample = pd.concat(data).query('Sample == "Full_Sample"')
        final_output_full_sample['Robust'] = ""

        final_output = (pd.concat([final_output,final_output_full_sample])
                        .sort_values(['var', 'Sample'])
                        .reset_index(drop=True))

        return final_output

    def run_sample_anovas(data, df1_ids_subs, df2_ids_subs, full_sample_idsub, anova_vars, formula):
    
        global demos_baseline

        #full_sample_idsub = pd.concat([df1_ids_subs, df2_ids_subs])
        
        from timeit import default_timer as timer
        start = timer()

        sample_tests = []
        for i,j in zip([df1_ids_subs, df2_ids_subs, full_sample_idsub], ['Sample1', 'Sample2', 'Full_Sample']):
            test = pd.merge(i, data, on = 'ID')
            test = (pd.merge(test, demos_baseline[['ID', 'abcd_site']], on ='ID')
                    .drop_duplicates('ID'))
            sample_anova = run_anova(test, anova_vars, formula)
            sample_anova['Sample'] = j
            sample_tests.append(sample_anova)

        output = format_sample_anovas(sample_tests)
        end = timer()
        perc_significant = (output.query('Robust == "True"').shape[0]/2)/len(anova_vars) 

        #print('Time to completion:', round(end - start,2))
        #print('Percent Robustly Significant:', round(perc_significant,2))
        return output
    
    def correct_pvals(data, sample):

        sample_data = data.query('Sample =='+'"'+sample+'"').reset_index(drop=True)
        pvals = sample_data.pval

        import statsmodels 

        fdr_pvals = (pd.DataFrame(
            statsmodels.stats.multitest.fdrcorrection(pvals, alpha=0.05)[1], columns = ['FDR_pval'])
                     .reset_index(drop=True)
                    )
        
        sample_data = pd.concat([sample_data, fdr_pvals], axis=1)
        sample_data['FDR_pval'] = sample_data['FDR_pval']
        sample_data = sample_data[['Sample', 'var', 'fval', 'pval', 'FDR_pval', 'Sig_Post', 'Robust']]

        return sample_data.round(3)

    def correct_output(output):

        fdr_list = []
        for i in output.Sample.unique():
            fdr_list.append(correct_pvals(output, i))

        final = pd.concat(fdr_list).sort_values(['var', 'Sample'])

        return final

    def common_values(list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        return list(set1.intersection(set2))

    def remove_duplicates(input_list):
        return list(set(input_list))

    def balance_samples(data):
        
        #if len(data.Subtype.unique()) == 5:
         #   data2 = data.query('Subtype == 5')
         #   data = data.query('Subtype != 5')
            
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler()
        X_rus, y_rus = rus.fit_resample(np.array(data), data.Subtype)

        balanced = pd.DataFrame(X_rus)
        balanced.columns = data.columns

        #if len(data.Subtype.unique()) == 5:
            #num = int(balanced.groupby('Subtype').count().reset_index().ID[0])
            #resampled_data = data2.sample(n=num, replace=True, random_state=42)
        #    balanced=pd.concat([balanced, data2])
            
        return balanced

    bootstrapped_cbcl_list = []
    bootstrapped_cog_list = []
    bootstrapped_ef_list = []
    bootstrapped_upps_list = []
    bootstrapped_adversity_list = []
    bootstrapped_stroop_list = []
    bootstrapped_nback_list = []
    #bootstrapped_matrix_list =[]
    
    for i in range(n_boots):
        # performing Stratified Sampling With Pandas and Numpy

        import pandas as pd
        import numpy as np

        import numpy
        from sklearn.utils import resample

        # lets say we wanted to shrink our data frame down to 125 rows, 
        # with the same
        # target class distribution
        
        #size = round(df1_rest.shape[0] * 2/3)
        
        df1_copy = df1_idsub.copy()
        df2_copy = df2_idsub.copy()
        full_sample_copy = full_sample_idsub.copy()
        
        #df1_balanced = balance_samples(df1_copy)
        #df2_balanced = balance_samples(df2_copy)
        
        df1_balanced = df1_copy
        df2_balanced = df2_copy
        full_sample_balanced = full_sample_idsub.copy()

        stratified_s1 = resample(df1_balanced, 
                                 #n_samples=round(df1_rest.shape[0]*(2/3)), 
                   replace=True, 
                                 #stratify=df1_balanced.Subtype
                                )

        stratified_s2 = resample(df2_balanced, 
                                 #n_samples=round(df2_rest.shape[0]*(2/3)), 
                   replace=True, 
                                 #stratify=df2_balanced.Subtype
                                )
        
        stratified_full = resample(full_sample_balanced, 
                         #n_samples=round(df2_rest.shape[0]*(2/3)), 
                   replace=True, 
                                   #stratify=full_sample_balanced.Subtype
                                  )

        df1_resampled = stratified_s1[['ID', 'Subtype']]
        df2_resampled = stratified_s2[['ID', 'Subtype']]
        full_sample_resampled = stratified_full[['ID', 'Subtype']]

        formula = '~ C(Subtype) + C(abcd_site)'
        
        global beh_all,beh_all_names, stroop_beh, cog_all,cog_all_names, demos_basline, nback, upps_factors, cog_ef_factors
        #global cbcl_r_vars, cbcl_base_t, nback, pcs, stroop_beh, upps_factors, demos_adv, matrix
        
        #CBCL ---------------------------------------------------------------------------------------------------------
        cbcl_anovas = (correct_output(run_sample_anovas(beh_all, df1_resampled, df2_resampled, full_sample_resampled, beh_all_names,formula))
                       .drop('pval', axis=1)
                      .rename({'FDR_pval':'pval'}, axis=1))
        cbcl_anovas['Measure'] = 'cbcl'
        bootstrapped_cbcl_list.append(cbcl_anovas)
        
        # EF Factors ---------------------------------------------------------------------------------------------------
        #ef_vars = list(cog_ef_factors.iloc[:, 1:].columns)
        ef_vars = list(cog_ef_factors.iloc[:, 1:].columns)
        cog_ef_all = pd.merge(cog_ef_factors, cog_all, on = ['ID'])[['ID'] + ef_vars]
    
        ef_anovas = (correct_output(run_sample_anovas(cog_ef_all, df1_resampled, df2_resampled, full_sample_resampled, ef_vars,formula))  
                      .drop('pval', axis=1)
                      .rename({'FDR_pval':'pval'}, axis=1))
        ef_anovas['Measure'] = 'ef'
        bootstrapped_ef_list.append(ef_anovas)
        
        #NBACK ---------------------------------------------------------------------------------------------------------
        nback_vars = list(nback.iloc[:, 1:].columns)
        nback_anovas = (correct_output(run_sample_anovas(nback, df1_resampled, df2_resampled, full_sample_resampled, nback_vars, formula))
                        .drop('pval', axis=1)
                      .rename({'FDR_pval':'pval'}, axis=1))
        nback_anovas['Measure'] = 'nback'
        bootstrapped_nback_list.append(nback_anovas)
        
        #NIH -----------------------------------------------------------------------------------------------------------
        #cog_vars = list(pcs.iloc[:, 1:].columns)
        cog_anovas = (correct_output(run_sample_anovas(cog_all, df1_resampled, df2_resampled, full_sample_resampled, cog_all_names,formula))
                      .drop('pval', axis=1)
                      .rename({'FDR_pval':'pval'}, axis=1))
        cog_anovas['Measure'] = 'cog'
        bootstrapped_cog_list.append(cog_anovas)
        
        #STROOP --------------------------------------------------------------------------------------------------------
        #stroop_beh_vars = list(stroop_beh.iloc[:, 1:].columns)
        stroop_beh_vars = ['Stroop_interf_acc_all_r','Happy_Acc_Eq_r','Angry_Acc_Eq_r']
        stroop_beh_anovas = (correct_output(run_sample_anovas(stroop_beh, df1_resampled, df2_resampled, full_sample_resampled, stroop_beh_vars,formula))
                             .drop('pval', axis=1)
                      .rename({'FDR_pval':'pval'}, axis=1))
        stroop_beh_anovas['Measure'] = 'stroop'
        bootstrapped_stroop_list.append(stroop_beh_anovas)
        
        #UPPS -----------------------------------------------------------------------------------------------------------
        upps_vars = list(upps_factors.iloc[:, 1:].columns)
        upps_anovas = (correct_output(run_sample_anovas(upps_factors, df1_resampled, df2_resampled, full_sample_resampled, upps_vars, formula))
                       .drop('pval', axis=1)
                      .rename({'FDR_pval':'pval'}, axis=1))
        upps_anovas['Measure'] = 'upps'
        bootstrapped_upps_list.append(upps_anovas)

        #Adversity -------------------------------------------------------------------------------------------------------
        adversity_demos=demos_baseline[['ID', 'adversity']]
        adversity_anovas = run_sample_anovas(adversity_demos, df1_resampled, df2_resampled, full_sample_resampled, 'adversity', formula)
        adversity_anovas['Measure'] = "adversity" 
        bootstrapped_adversity_list.append(adversity_anovas)

        #Matrix -------------------------------------------------------------------------------------------------------
        #matrix_anovas = run_sample_anovas(matrix, df1_resampled, df2_resampled,'pea_wiscv_tss_r', formula)
        #matrix_anovas['Measure'] = "matrix" 
        #bootstrapped_matrix_list.append(matrix_anovas)
        
    
    bootstrapped_cbcl_df = pd.concat(bootstrapped_cbcl_list).sort_values('Sample', ascending=False)
    bootstrapped_cog_df = pd.concat(bootstrapped_cog_list).sort_values('Sample', ascending=False)
    bootstrapped_upps_df = pd.concat(bootstrapped_upps_list).sort_values('Sample', ascending=False)
    bootstrapped_adversity_df = pd.concat(bootstrapped_adversity_list).sort_values('Sample', ascending=False)
    bootstrapped_stroop_df = pd.concat(bootstrapped_stroop_list).sort_values('Sample', ascending=False)
    bootstrapped_nback_df = pd.concat(bootstrapped_nback_list).sort_values('Sample', ascending=False)
    bootstrapped_ef_df = pd.concat(bootstrapped_ef_list).sort_values('Sample', ascending=False)
    #bootstrapped_matrix_df = pd.concat(bootstrapped_matrix_list)
 
    def show_point(data, og_data, xvar, xlab, labs=None):
        
        n_vars = int(len(data['var'].unique()))
        #fig_size = (5, 0.5*n_vars)
        import seaborn as sns
        plt.figure(figsize=(5, 0.5*n_vars)) #10

        ax = sns.pointplot(y="var", x=xvar,
                    hue="Sample", palette=["#808080", "#4645E2", "#2CE26F"],
                    data=data,
                    dodge=0.4, join=False)

        sns.pointplot(y="var", x=xvar,
                    hue="Sample", palette=["black", "black", "black"],
                    data=og_data,
                    scale = .5,
                    dodge=0.4, join=False)


        ax.axvline(0.05, linewidth=1, color='red', linestyle = 'dashed')

        if labs is not None:
            ax.set_yticklabels(labs)

        ax.tick_params(axis='x', rotation=0)

        ax.set(xlabel=xlab, ylabel='')

        legend_handles, _= ax.get_legend_handles_labels()

        ax.legend(legend_handles, ['Full Sample', 'Sample 1', 'Sample 2'],
                  bbox_to_anchor=(1, 1.05), borderaxespad=0, ncol=3, frameon=False)

        plt.tight_layout()
        
        return ax

    def boot_msd(data):
        msd = (data[['var', 'fval', 'pval', 'Sample']]
         .groupby(['Sample', 'var'])
         .describe()
         .T
         .reset_index(level=0, drop=True)
         .T
         .reset_index()
         .loc[:, list(itertools.chain.from_iterable([['Sample', 'var'], ['mean', 'std']]))]
        )

        msd.columns = ['Sample', 'var', 'f_mean', 'p_mean', 'f_std', 'p_std']

        return msd

    def get_robust_index(data):

        names = data.groupby('var').count().reset_index().reset_index()

        robust_names = (data
                       .query('p_mean < .05')
                       .groupby('var')
                       .count()
                       .reset_index()
                       .query('p_mean == 3')
                      )

        robust_list = pd.merge(names, robust_names, on='var')['index']

        return robust_list
    
    def set_size(w,h, ax=None):
        """ w, h: width, height in inches """
        if not ax: ax=plt.gca()
        l = ax.figure.subplotpars.left
        r = ax.figure.subplotpars.right
        t = ax.figure.subplotpars.top
        b = ax.figure.subplotpars.bottom
        figw = float(w)/(r-l)
        figh = float(h)/(t-b)
        ax.figure.set_size_inches(figw, figh)


    def get_robust_outputs(data, og_data, labs=None, save=None):
        
        n_vars = int(len(data['var'].unique()))
        import seaborn as sns
        plt.figure(figsize=(5, 0.5*n_vars)) #10

        bootstrapped_msd = boot_msd(data)
        
        
        g = show_point(data, og_data, 'pval', 'Corrected P-Value', labs)

        for i in get_robust_index(bootstrapped_msd):
            g.axhline(i, linewidth=45, color='red', alpha = .1)
            # Set the x-axis limits
        
        plt.xlim(0, 1)
        set_size(3,n_vars)
        
        plt.tight_layout()
        plt.clf()
        if save is not None:
            g.savefig(save, dpi=300)
            plt.clf()

        return g,  bootstrapped_msd


    #cbcl_labs = ['Aggressive', 'Anxious Depressed', 'Attention Problems',
    #             'Externalizing', 'Internalizing', 'Rule Breaking',
    #             'Social Problems', 'Somatic Complaints', 'Thought Problems',
    #             'Total Problems', 'Withdrawn Depressed']

    #upps_labs = bootstrapped_upps_df['var'].str.replace("_", " ").unique()

    #stroop_pull = ['Angry_Acc_Eq', 'Angry_RT_Eq', 'HappyMinusAngry_Acc_Eq',
    #'HappyMinusAngry_RT_Eq', 'Happy_Acc_Eq', 'Happy_RT_Eq',
    #'Stroop_interf_RT_Eq', 'Stroop_interf_RT_MC',
    #'Stroop_interf_RT_all', 'Stroop_interf_RT_angryMC',
    #'Stroop_interf_RT_happyMC', 'Stroop_interf_acc_Eq',
    #'Stroop_interf_acc_MC', 'Stroop_interf_acc_all',
    #'Stroop_interf_acc_angryMC', 'Stroop_interf_acc_happyMC']

    #bootstrapped_stroop_df_filtered = bootstrapped_stroop_df.query('var in @stroop_pull')

    stroop_labs = (bootstrapped_stroop_df['var']
                   .str.replace('Stroop_','')
                   .str.replace('_',' ').unique()
                  )

    #bootstrapped_stroop_df_filtered = bootstrapped_stroop_df#.query('var in @stroop_pull')

    #bootstrapped_robust_msd_outputs[0]
    og_anovas = pd.read_csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/group_differences/rest_dti_smri_anova/'+data_type+'_anova.csv')
    og_cog_anovas =  og_anovas.query('Measure == "cog"').drop('pval', axis=1).rename({'FDR_pval':'pval'}, axis=1).sort_values('Sample', ascending=False)
    og_adversity_anovas = og_anovas.query('Measure == "adversity"').sort_values('Sample', ascending=False)
    og_nback_anovas = og_anovas.query('Measure == "nback"').drop('pval', axis=1).rename({'FDR_pval':'pval'}, axis=1).sort_values('Sample', ascending=False)
    og_stroop_anovas = og_anovas.query('Measure == "stroop"').drop('pval', axis=1).rename({'FDR_pval':'pval'}, axis=1).sort_values('Sample', ascending=False)
    og_upps_anovas = og_anovas.query('Measure == "upps"').drop('pval', axis=1).rename({'FDR_pval':'pval'}, axis=1).sort_values('Sample', ascending=False)
    og_cbcl_anovas = og_anovas.query('Measure == "cbcl"').drop('pval', axis=1).rename({'FDR_pval':'pval'}, axis=1).sort_values('Sample', ascending=False)
    og_ef_anovas = og_anovas.query('Measure == "ef"').drop('pval', axis=1).rename({'FDR_pval':'pval'}, axis=1).sort_values('Sample', ascending=False)
    #og_matrix_anovas = og_anovas.query('Measure == "matrix"').drop('pval', axis=1).rename({'FDR_pval':'pval'}, axis=1)

    bootstrapped_robust_fig_outputs = []
    bootstrapped_robust_msd_outputs = []

    full_df = pd.concat([bootstrapped_cog_df, bootstrapped_cbcl_df,bootstrapped_adversity_df, bootstrapped_stroop_df, bootstrapped_nback_df, bootstrapped_upps_df, bootstrapped_ef_df])
    full_anovas = pd.concat([og_cog_anovas, og_cbcl_anovas, og_adversity_anovas, og_stroop_anovas, og_nback_anovas, og_upps_anovas, og_ef_anovas])
    
    for i,o,j in zip(
        
        [bootstrapped_cog_df, bootstrapped_cbcl_df,bootstrapped_adversity_df, bootstrapped_stroop_df, bootstrapped_nback_df, bootstrapped_upps_df, bootstrapped_ef_df, full_df],
            
        [og_cog_anovas, og_cbcl_anovas, og_adversity_anovas, og_stroop_anovas, og_nback_anovas, og_upps_anovas, og_ef_anovas, full_anovas],    

        [None, None, None, stroop_labs, None, None, None]):

        bootstrapped_fig, bootstrapped_df_msd = get_robust_outputs(i, o, j)
        bootstrapped_robust_fig_outputs.append(bootstrapped_fig)
        bootstrapped_robust_msd_outputs.append(bootstrapped_df_msd)

    for i,j in zip(bootstrapped_robust_fig_outputs, 
                   ['cog', 'cbcl', 'adversity', 'stroop', 'nback', 'upps', 'ef', 'full']):
        fig_path= '/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/group_differences/bootstrapped_outputs/anova/figures/'
        i.figure.savefig(fig_path+j+'_'+data_type+'.png')


    for i,j in zip(bootstrapped_robust_msd_outputs, 
                   ['cog', 'cbcl', 'adversity', 'stroop', 'nback', 'upps', 'ef', 'full']):
        msd_path= '/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/group_differences/bootstrapped_outputs/anova/msd/'
        i.round(3).to_csv(msd_path+j+'_'+data_type+'.csv')

    def format_msd(data):
        data['F'] = data.f_mean.astype(str)+' ('+data.f_std.astype(str)  + ')'
        data['P'] = data.p_mean.astype(str)+' ('+data.p_std.astype(str) + ')'
        data = (data.loc[:, ['Sample', 'var', 'F', 'P']]
                .pivot(index='var', columns='Sample')
               )

        return data

    msd_paths = f'/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/group_differences/bootstrapped_outputs/anova/msd/*_{data_type}'

    from glob import glob
    msd_outputs = []
    for i in sorted(glob(msd_paths, recursive = True)):
        formatted_msd = format_msd(pd.read_csv(i).iloc[:, 1:])
        formatted_msd.to_csv(i)
        msd_outputs.append(formatted_msd)
    
    return full_df, full_anovas


# In[2]:


def show_point(data, og_data, xvar, xlab, labs=None):

    n_vars = int(len(data['var'].unique()))
    #fig_size = (5, 0.5*n_vars)
    import seaborn as sns
    plt.figure(figsize=(5, 0.5*n_vars)) #10

    ax = sns.pointplot(y="var", x=xvar,
                hue="Sample", palette=["#808080", "#4645E2", "#2CE26F"],
                data=data,
                dodge=0.4, join=False)

    sns.pointplot(y="var", x=xvar,
                hue="Sample", palette=["black", "black", "black"],
                data=og_data,
                scale = .5,
                dodge=0.4, join=False)


    ax.axvline(0.05, linewidth=1, color='red', linestyle = 'dashed')

    if labs is not None:
        ax.set_yticklabels(labs)

    ax.tick_params(axis='x', rotation=0)

    ax.set(xlabel=xlab, ylabel='')

    legend_handles, _= ax.get_legend_handles_labels()

    ax.legend(legend_handles, ['Full Sample', 'Sample 1', 'Sample 2'],
              bbox_to_anchor=(1, 1.05), borderaxespad=0, ncol=3, frameon=False)

    plt.tight_layout()

    return ax

def boot_msd(data):
    msd = (data[['var', 'fval', 'pval', 'Sample']]
     .groupby(['Sample', 'var'])
     .describe()
     .T
     .reset_index(level=0, drop=True)
     .T
     .reset_index()
     .loc[:, list(itertools.chain.from_iterable([['Sample', 'var'], ['mean', 'std']]))]
    )

    msd.columns = ['Sample', 'var', 'f_mean', 'p_mean', 'f_std', 'p_std']

    return msd

def get_robust_index(data):
    
    #data = data.query('Sample != "Full Sample"')

    #plot_pull = ['CommonEF', 'Intelligence', 'UpdatingSpecific', 'attention_problems_r',
    #             'ext_factor', 'int_factor', 'p_factor', 'social_problems_r',
    #             'thought_problems_r', 'negative_urgency', 'perserverance',
    #             'positive_urgency', 'predmeditation', 'sensation_seeking']
    
    plot_pull = ['CommonEF', 'Intelligence', 'UpdatingSpecific', 'pc1_new_r', 'pc2_new_r', 'pc3_new_r', 'LMT_r', 'RAVLT_r', 'agressive_r', 'anxious_depressed_r', 
             'attention_problems_r','externalizing_r', 'internalizing_r', 'rule_breaking_r','social_problems_r', 
             'somatic_complaints_r', 'thought_problems_r','total_r', 'withdrawn_depressed_r', 
             'positive_urgency', 'negative_urgency', 'perserverance','predmeditation', 'sensation_seeking']

    names = data.groupby('var').count().sort_values(by='var', key=lambda x: x.map({v: i for i, v in enumerate(plot_pull)})).reset_index().reset_index()

    robust_names = (data
                   .query('p_mean < .05')
                   .groupby('var')
                   .count()
                   .reset_index()
                   .query('p_mean == 3')
                   #.query('p_mean == 2')
                  )

    robust_list = pd.merge(names, robust_names, on='var')['index']

    return robust_list

def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


def get_robust_outputs(data, og_data, save):
    
    n_vars = int(len(data['var'].unique()))
    import seaborn as sns
    plt.figure(figsize=(5, 0.5*n_vars)) #10

    bootstrapped_msd = boot_msd(data)


    g = show_point(data, og_data, 'pval', 'Corrected P-Value', None)

    for i in get_robust_index(bootstrapped_msd):
        g.axhline(i, linewidth=45, color='red', alpha = .1)
        # Set the x-axis limits

    plt.xlim(0, 1)
    set_size(3,n_vars)

    plt.tight_layout()

    #if save is not None:
    g.figure.savefig(save, dpi=300)

    #return g,  bootstrapped_msd


# In[8]:


plot_pull = ['CommonEF', 'Intelligence', 'UpdatingSpecific', 'LMT_r', 'RAVLT_r', 'agressive_r', 'anxious_depressed_r', 
             'attention_problems_r','externalizing_r', 'internalizing_r', 'rule_breaking_r','social_problems_r', 
             'somatic_complaints_r', 'thought_problems_r','total_r', 'withdrawn_depressed_r', 
             'positive_urgency', 'negative_urgency', 'perserverance','predmeditation', 'sensation_seeking', 
             'pc1_new_r', 'pc2_new_r', 'pc3_new_r']


# In[76]:


full_df_rest_include, full_anovas_rest_include = boot_pipeline(
    sample1_rest_include_idsub, sample2_rest_include_idsub, full_sample_rest_include_idsub, 'rest_include', 1000)


full_df_rest_pulled_include = (full_df_rest_include
                  .query('var in @plot_pull')
                  .sort_values(by='var', key=lambda x: x.map({v: i for i, v in enumerate(plot_pull)})))

full_anovas_rest_pulled_include = (full_anovas_rest_include
                      .query('var in @plot_pull')
                      .sort_values(by='var', key=lambda x: x.map({v: i for i, v in enumerate(plot_pull)})))

get_robust_outputs(full_df_rest_pulled_include, full_anovas_rest_pulled_include, 
                   save='/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/group_differences/bootstrapped_outputs/anova/figures/paper_rest_include.png')


# In[77]:


full_df_rest_combined, full_anovas_rest_combined = boot_pipeline(sample1_rest_combined_idsub, sample2_rest_combined_idsub, full_sample_rest_combined_idsub, 'rest_combined', 1000)

full_df_rest_pulled_combined = (full_df_rest_combined
                  .query('var in @plot_pull')
                  .sort_values(by='var', key=lambda x: x.map({v: i for i, v in enumerate(plot_pull)})))

full_anovas_rest_pulled_combined = (full_anovas_rest_combined
                      .query('var in @plot_pull')
                      .sort_values(by='var', key=lambda x: x.map({v: i for i, v in enumerate(plot_pull)})))

get_robust_outputs(full_df_rest_pulled_combined, full_anovas_rest_pulled_combined, 
                   save='/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/group_differences/bootstrapped_outputs/anova/figures/paper_rest_combined.png')


# In[13]:


full_df_rest_dont_include, full_anovas_rest_dont_include = boot_pipeline(full_sample_rest_dont_include_idsub, 
                                                                         full_sample_rest_dont_include_idsub, 
                                                                         full_sample_rest_dont_include_idsub, 'rest_dont', 1000)

full_df_rest_pulled_dont_include = (full_df_rest_dont_include
                                .query('var in @plot_pull')
                                .sort_values(by='var', key=lambda x: x.map({v: i for i, v in enumerate(plot_pull)})))

full_anovas_rest_pulled_dont_include = (full_anovas_rest_dont_include
                                    .query('var in @plot_pull')
                                    .sort_values(by='var', key=lambda x: x.map({v: i for i, v in enumerate(plot_pull)})))

get_robust_outputs(full_df_rest_pulled_dont_include, full_anovas_rest_pulled_dont_include, 
                   save='/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/group_differences/bootstrapped_outputs/anova/figures/paper_rest_dont.png')


# In[78]:


#!jupyter nbconvert --to script boostrapped_anova_pipeline.ipynb


# In[ ]:





# In[ ]:





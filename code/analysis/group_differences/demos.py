#!/usr/bin/env python
# coding: utf-8

# In[273]:


import sys
sys.path.append('/pl/active/banich/studies/Relevantstudies/abcd/env/lib/python3.7/site-packages')
sys.path.append('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/')

from import_data import *
from import_subtypes import *

# In[262]:

def demo_pipeline(full_sample, sample1, sample2, datatype):
    
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

    def run_sample_anovas(data, anova_vars, formula, full_sample_idsub, df1_ids_subs, df2_ids_subs):

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

    def chitest(data, group, var):
        data = data[[group, var]].dropna()
        data_crosstab =pd.crosstab(data[group], data[var], margins = False)
        array_crosstab = pd.DataFrame(np.array(data_crosstab))
        chi = pd.DataFrame(np.array(chi2_contingency(data_crosstab))).T[[0, 1]].rename(columns={0: 'stat', 1: 'pval'})
        chi['var'] = var

        return chi[['var', 'stat', 'pval']]

    # In[263]:
    
    sample1_demos = (sample1
     .merge(demos_baseline.assign(age = demos_baseline['age']/12), on='ID', how='outer')
     .drop_duplicates('ID', keep='first')
     .drop('adversity', axis=1)
     .dropna(subset=['Subtype'])
     #.query('abcd_site != "site22"')
    )

    sample2_demos = (sample2
     .merge(demos_baseline.assign(age= demos_baseline['age']/12), on='ID', how='outer')
     .drop_duplicates('ID', keep='first')
     .drop('adversity', axis=1)
     .dropna(subset=['Subtype'])
     #.query('abcd_site != "site22"')
    )

    full_sample_demos = (full_sample
     .merge(demos_baseline.assign(age= demos_baseline['age']/12), on='ID', how='outer')
     .drop_duplicates('ID', keep='first')
     .drop('adversity', axis=1)
     .dropna(subset=['Subtype'])
     #.query('abcd_site != "site22"')
    )

    demo_cat_vars = sample1_demos.drop(['ID', 'age', 'Subtype', 'abcd_site', 'hisp'], axis=1).columns


    # In[265]:


    def chitest(data, group, var):
        data = data[[group, var]].dropna()
        data_crosstab =pd.crosstab(data[group], data[var], margins = False)
        array_crosstab = pd.DataFrame(np.array(data_crosstab))
        chi = pd.DataFrame(np.array(chi2_contingency(data_crosstab))).T[[0, 1]].rename(columns={0: 'stat', 1: 'pval'})
        chi['var'] = var

        return chi[['var', 'stat', 'pval']]


    # In[266]:


    def chi_test_wrapper(df_list, var_list):

        def chitest(data, group, var):

            data = data[[group, var]].dropna()
            data_crosstab = pd.crosstab(data[group], data[var], margins=False)
            array_crosstab = pd.DataFrame(np.array(data_crosstab))
            chi = pd.DataFrame(np.array(chi2_contingency(data_crosstab))).T[[0, 1]].rename(columns={0: 'stat', 1: 'pval'})
            chi['var'] = var
            return chi[['var', 'stat', 'pval']]

        chi_result = map(lambda var: pd.concat(map(lambda df: chitest(df, 'Subtype', var), df_list)), var_list)


        return pd.concat(chi_result)


    # In[267]:


    chi_tests = chi_test_wrapper([full_sample_demos, sample1_demos, sample2_demos], demo_cat_vars)
    chi_tests['Sample'] = ['Full_Sample', 'Sample1', 'Sample2']* int(chi_tests.shape[0]/3)


    # In[268]:


    def subgroup_counts(df, subtype_col, count_col):
            # Group the dataframe by subtype_col and count the occurrences in count_col
            group_counts = df.groupby(subtype_col)[count_col].value_counts()
            # Convert to dataframe and reset index
            group_counts_df = group_counts.to_frame(name='Count').reset_index()
            # Calculate percentage
            #group_counts_df['Percentage'] = group_counts_df['Count'] / group_counts_df['Count'].sum() * 100
            percs = []
            for i in group_counts_df[subtype_col].unique():
                sub = group_counts_df.query('Subtype =='+str(i))
                sub = sub.assign(Percentage = (sub['Count'] / sub['Count'].sum())*100)   
                sub = sub.assign(Summary = sub['Count'].astype(str) + " ("+round(sub['Percentage'],2).astype(str)+"%)")
                sub = sub.drop(['Count', 'Percentage'],axis=1)
                percs.append(sub)

            output=pd.concat(percs).reset_index(drop=True)
            return output

    sub_percs = []
    for x in demo_cat_vars: 
        for i,j in zip([full_sample_demos, sample1_demos, sample2_demos], ['Full_Sample', 'Sample1', 'Sample2']):
            output = subgroup_counts(i, 'Subtype', x)
            output['Sample'] = j
            sub_percs.append(output)

    sub_percs = pd.concat(sub_percs).melt(id_vars=['Subtype', 'Sample', 'Summary']).dropna()
    sub_percs = sub_percs.pivot(index=['variable','value'], columns=['Sample', 'Subtype'], values='Summary')


    # In[269]:


    age_anovas = (run_sample_anovas(demos_baseline, 'age', '~C(Subtype)', full_sample, sample1, sample2)
     .drop(['Robust'], axis=1)).rename({'fval':'stat'},axis=1)


    # In[288]:


    tests = pd.concat([age_anovas, chi_tests])
    tests.stat = tests.stat.astype('float').round(2)
    tests.pval = tests.pval.astype('float')

    def format_pval(val):
        if val >= 0.05:
            return ""
        elif val >= 0.01:
            return "*"
        elif val >= 0.001:
            return "**"
        else:
            return "***"

    tests['pval'] = tests['pval'].apply(format_pval)

    tests = tests.assign(F_X2 = tests.stat.astype(str) + tests.pval.astype(str))[['Sample', 'var', 'F_X2']]
    tests = tests.pivot(index='var', columns='Sample', values='F_X2')


    # In[274]:


    def prop_test(data, group, var):

        import numpy as np

        data = data[[group, var]].dropna()
        data_crosstab = pd.crosstab(data[group], data[var], margins=False)
        data_crosstab['total'] = data_crosstab.sum(axis=1)

        var_cat_list = [s for s in data_crosstab.columns.to_list() if s != 'total']
        sub_list = data_crosstab.index.to_list()


        import itertools
        # Create a list
        lst = [int(x) for x in data_crosstab.index.to_list()]   
        # Generate all possible pairs
        pairs = list(itertools.combinations(lst, 2))
        # Filter out duplicate pairs
        unique_pairs = list(set([tuple(sorted(pair)) for pair in pairs]))

        post_hoc_z = []
        for i in unique_pairs:
            for j in var_cat_list:
                count = np.array(data_crosstab.query('index == '+str(i[0]) + 'or index =='+str(i[1]))[[j]].iloc[:, 0]) 
                nobs = np.array(data_crosstab.query('index == '+str(i[0]) + 'or index =='+str(i[1]))[['total']].iloc[:, 0]) 
                import numpy as np
                from statsmodels.stats.proportion import proportions_ztest
                stat, pval = proportions_ztest(count, nobs)


                if count[0] > count[1]:
                    result = f'{i[0]} > {i[1]}'
                else:
                    result = f'{i[0]} < {i[1]}'

                output = pd.DataFrame([j, i, pval, result]).T
                output.columns = ['cat', 'pair', 'pval', 'post_hoc']

                post_hoc_z.append(output)

        post_hoc_z = pd.concat(post_hoc_z).sort_values(['cat', 'pair']).query('pval < .05')

        final = (pd.DataFrame(post_hoc_z[['cat', 'post_hoc']]
                              .groupby('cat')['post_hoc'].agg(lambda x: ', '.join(x)))
                 .reset_index()
                 .assign(var = var))[['var', 'cat', 'post_hoc']]

        return final 


    # In[285]:


    prop_list=[]
    for i,j in zip([full_sample_demos, sample1_demos, sample2_demos],  ['Full_Sample', 'Sample1', 'Sample2']):
        for k in demo_cat_vars:
            prop_list.append(prop_test(i, 'Subtype', k).assign(Sample = j))

    prop_tests = pd.concat(prop_list).pivot(index=['var', 'cat'], columns='Sample', values='post_hoc')  


    # In[351]:


    full_sample_perc = pd.concat([sub_percs['Full_Sample'], prop_tests['Full_Sample']], axis=1).assign(Sample = 'Full Sample')
    sample1_perc = pd.concat([sub_percs['Sample1'], prop_tests['Sample1']], axis=1).assign(Sample = 'Sample 1')
    sample2_perc = pd.concat([sub_percs['Sample2'], prop_tests['Sample2']], axis=1).assign(Sample = 'Sample 2')

    sub_cols = ['Subtype ' + str(value) for value in sorted(list(sample1.Subtype.unique()))]+ ['Post Hoc', 'Sample']

    for i in full_sample_perc, sample1_perc, sample2_perc:
        i.columns = sub_cols


    # In[328]:


    def msd(data, group, column): 

        grouped_data = data[[group] + [column]]
        grouped_data.iloc[:, 1:] = grouped_data.iloc[:, 1:].apply(pd.to_numeric)
        grouped_data = grouped_data.groupby(group).agg(['mean', 'std']).reset_index()

        col_name = grouped_data.columns[1][0] + "_msd"

        grouped_data[col_name] = grouped_data.iloc[:, 1].apply(lambda x: f'{x:.2f}') +                              grouped_data.iloc[:, 2].apply(lambda x: f' ({x:.2f})')

        grouped_data = pd.DataFrame(grouped_data[[col_name]].T)
        # Rename the columns
        grouped_data.columns = [f'{group} {i+1}' for i in range(len(grouped_data.columns))]
        grouped_data = grouped_data.reset_index(drop=True)
        grouped_data = grouped_data.rename(index={0: column})

        return grouped_data


    # In[403]:


    age_list=[]
    for i,j in zip([full_sample_demos, sample1_demos, sample2_demos],
                   ['Full_Sample', 'Sample1', 'Sample2']):
        sig_post = age_anovas.query('Sample =='+'"'+j+'"')[['Sig_Post']].iloc[0][0]
        age_list.append(msd(i, 'Subtype', 'age').assign(post = sig_post, Sample = j))

    age_out = pd.concat(age_list)
    age_out.columns = sub_cols

    def multi(df):
        df.index = pd.MultiIndex.from_product([['age'], df.index])
        return df

    full_sample_age = multi(age_out.query('Sample == "Full_Sample"'))
    sample1_age = multi(age_out.query('Sample == "Sample1"'))
    sample2_age = multi(age_out.query('Sample == "Sample2"'))


    # In[406]:


    full_sample_final = pd.concat([full_sample_age, full_sample_perc]).fillna('')
    sample1_final = pd.concat([sample1_age, sample1_perc]).fillna('')
    sample2_final = pd.concat([sample2_age, sample2_perc]).fillna('')


    # In[457]:


    outpath = '/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/group_differences/demos_outputs/'+datatype+'/'
    tests.to_csv(outpath+datatype+'_tests.csv')
    full_sample_final.to_csv(outpath+datatype+'_full_sample_table.csv')
    sample1_final.to_csv(outpath+datatype+'_sample1_table.csv')
    sample2_final.to_csv(outpath+datatype+'_sample2_table.csv')



demo_pipeline(full_sample_rest_include_idsub, sample1_rest_include_idsub, sample2_rest_include_idsub, 'rest_include')
demo_pipeline(full_sample_rest_combined_idsub, sample1_rest_combined_idsub, sample2_rest_combined_idsub, 'rest_combined')
#demo_pipeline(full_sample_rest_dont_include_idsub, full_sample_rest_dont_include_idsub, full_sample_rest_dont_include_idsub, 'rest_dont_include')



import sys 
sys.path.append('/pl/active/banich/studies/Relevantstudies/abcd/env/lib/python3.7/site-packages')

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import os
import numpy as np
import pandas as pd
import scipy.io
from scipy import stats
from sklearn.manifold import MDS
import scipy.spatial.distance as sp_distance
import matplotlib.pyplot as plt
import seaborn as sns 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import sys
import scikit_posthocs as sp
import itertools

import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
import itertools
from functools import reduce
import networkx as nx

from scipy.stats import chi2_contingency
import scipy.stats as stats


from sklearn.metrics import classification_report

def case_when(*args, default):
    return np.select(
        condlist = [args[i] for i in range(0, len(args), 2)],
        choicelist = [args[i] for i in range(1, len(args), 2)],
        default=default
    )
    
#import shap

#import xgboost
#import xgboost as xgb
#from xgboost.sklearn import XGBRegressor

import datetime
from numpy import mean
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scipy.io
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_moons
from sklearn.manifold import SpectralEmbedding
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def readit(path):
    #data = reduce_memory_usage(pd.concat(pd.read_csv(path, iterator=True, chunksize=100000), ignore_index=True))
    data = pd.concat(pd.read_csv(path, iterator=True, chunksize=100000), ignore_index=True)
    return data


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


def get_pairs(data1, data2, var):
    
    mat = getposthoc(data1, var)
    mat2 = getposthoc(data2, var)
    pairs = list(itertools.combinations(uni(data1.Subtype),2))
    
    combos = []
    combos2 = []
    for i in pairs:
        combos.append(mat[i[0]][i[1]-1])
        combos2.append(mat2[i[0]][i[1]-1])

    def process_pairs(combo_list, var, sample):
        combos = pd.DataFrame(combo_list)
        combos['pairs'] = pairs
        combos['var'] = var
        combos['sample'] = sample
        combos = combos[combos.apply(lambda x: x[0] < .05, axis=1)]
        combos.columns = ['pval', 'pairs', 'var', 'sample']
        combos = combos[['pairs', 'pval', 'var', 'sample']]
        return combos

    p1 = process_pairs(combos, var, 'sample1')
    p2 = process_pairs(combos2, var, 'sample2')

    dups = pd.concat([p1, p2])
    mask = dups.pairs.duplicated(keep=False)
    dups = dups[mask]
    
    return dups

def uni(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)

    return unique_list

def region_plot(data, brain_region, cmap, sample, outpath=None):
    
    region = data.filter(regex=brain_region)
    columns = region.columns.str.replace('nBack', '', regex=True)
    columns = columns.str.replace('_', ' ', regex=True)
    columns = columns.str.replace('contrast in', '', regex=True)
    columns = columns.str.replace('ROI', '', regex=True)
    columns = columns.str.replace('versus', 'vs', regex=True)
    mean_length = 20
    columns = ["\n".join(textwrap.wrap(i,mean_length)) for i in columns]
    region.columns = columns 
    cluster = data.filter(regex='Subtype')
    region = pd.concat([cluster, region], axis=1)
    region_long = pd.melt(region, id_vars=['Subtype'], value_vars=columns)    
    plt.style.use('fivethirtyeight')
    #sns.set_style("fivethirtyeight", {'axes.grid' : False})    
    plt.figure(figsize=(14,7))
    ax = sns.lineplot(data=region_long, x="variable", y="value",
                      hue="Subtype", palette=cmap)
    #plt.grid(b=None)
    #sns.despine(offset=1)
    ax.grid(axis='x')
    plt.xticks(rotation = 25, ha = 'right', rotation_mode='anchor')
    plt.legend(title = "Subtype", loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set(xlabel = sample)
    #plt.ylim((-.4, .4)) 
    plt.title(brain_region.capitalize())
    plt.tight_layout()
    ax.spines['bottom'].set_color('black')
    ax.xaxis.set_ticks_position('bottom') 
    if outpath is not None:
        brain_path = outpath+brain_region +'/'
        os.system(f'mkdir -p {brain_path}')
        plt.savefig(brain_path+'_'+sample+'_'+brain_region+'.png')
        

def filt_sub(data, sub):
    sub = data[data['Subtype'] == sub].drop(['ID', 'Subtype', 'Unnamed: 0.1'], axis =1)
    return sub

def readtxt(path):
    rm_quote = lambda x: x.replace('"', '')
    txt = pd.read_csv(path, sep='\t+').applymap(rm_quote).drop(labels=0, axis=0).replace(r'^\s*$', np.NaN, regex=True)
    txt = txt.rename(columns=rm_quote)
    return txt 

import matplotlib.pyplot as plt
from seaborn.categorical import _ViolinPlotter
import numpy as np


class _SinaPlotter(_ViolinPlotter):

    def __init__(self, x, y, hue, data, order, hue_order,
                 bw, cut, scale, scale_hue, gridsize,
                 width, inner, split, dodge, orient, linewidth,
                 color, palette, saturation,
                 violin_facealpha, point_facealpha):
        # initialise violinplot
        super(_SinaPlotter, self).__init__(
            x, y, hue, data, order, hue_order,
            bw, cut, scale, scale_hue, gridsize,
            width, inner, split, dodge, orient, linewidth,
            color, palette, saturation
        )

        # Set object attributes
        self.dodge = dodge
        # bit of a hack to set color alphas for points and violins
        self.point_colors = [(*color, point_facealpha) for color in self.colors]
        self.colors = [(*color, violin_facealpha) for color in self.colors]

    def jitterer(self, values, support, density):
        if values.size:
            max_density = np.interp(values, support, density)
            max_density *= self.dwidth
            low = 0 if self.split else -1
            jitter = np.random.uniform(low, 1, size=len(max_density)) * max_density
        else:
            jitter = np.array([])
        return jitter

    def draw_sinaplot(self, ax, kws):
        """Draw the points onto `ax`."""
        # Set the default zorder to 2.1, so that the points
        # will be drawn on top of line elements (like in a boxplot)
        for i, group_data in enumerate(self.plot_data):
            if self.plot_hues is None or not self.dodge:

                if self.hue_names is None:
                    hue_mask = np.ones(group_data.size, np.bool)
                else:
                    hue_mask = np.array([h in self.hue_names
                                         for h in self.plot_hues[i]], np.bool)
                    # Broken on older numpys
                    # hue_mask = np.in1d(self.plot_hues[i], self.hue_names)

                strip_data = group_data[hue_mask]
                density = self.density[i]
                support = self.support[i]

                # Plot the points in centered positions
                cat_pos = np.ones(strip_data.size) * i
                cat_pos += self.jitterer(strip_data, support, density)
                kws.update(color=self.point_colors[i])
                if self.orient == "v":
                    ax.scatter(cat_pos, strip_data, **kws)
                else:
                    ax.scatter(strip_data, cat_pos, **kws)

            else:
                offsets = self.hue_offsets
                for j, hue_level in enumerate(self.hue_names):
                    hue_mask = self.plot_hues[i] == hue_level
                    strip_data = group_data[hue_mask]
                    density = self.density[i][j]
                    support = self.support[i][j]
                    if self.split:
                        # Plot the points in centered positions
                        center = i
                        cat_pos = np.ones(strip_data.size) * center
                        jitter = self.jitterer(strip_data, support, density)
                        #cat_pos = cat_pos + jitter if j else cat_pos - jitter
                        kws.update(color=self.point_colors[j])
                        if self.orient == "v":
                            ax.scatter(cat_pos, strip_data, zorder=2, **kws)
                        else:
                            ax.scatter(strip_data, cat_pos, zorder=2, **kws)
                    else:
                        # Plot the points in centered positions
                        #center = i + offsets[j]
                        center = i
                        cat_pos = np.ones(strip_data.size) * center
                        cat_pos += self.jitterer(strip_data, support, density)
                        kws.update(color=self.point_colors[j])
                        if self.orient == "v":
                            ax.scatter(cat_pos, strip_data, zorder=2, **kws)
                        else:
                            ax.scatter(strip_data, cat_pos, zorder=2, **kws)

    def add_legend_data(self, ax, color, label):
        """Add a dummy patch object so we can get legend data."""
        # get rid of alpha band
        if len(color) == 4:
            color = color[:3]
        rect = plt.Rectangle([0, 0], 0, 0,
                             linewidth=self.linewidth / 2,
                             edgecolor=self.gray,
                             facecolor=color,
                             label=label)
        ax.add_patch(rect)

    def plot(self, ax, kws):
        """Make the sinaplot."""
        if kws.pop('violin', True):
            self.draw_violins(ax)
        elif self.plot_hues is not None:
            # we need to add the dummy box back in for legends
            for j, hue_level in enumerate(self.hue_names):
                self.add_legend_data(ax, self.colors[j], hue_level)
        self.draw_sinaplot(ax, kws)
        self.annotate_axes(ax)
        if self.orient == "h":
            ax.invert_yaxis()


def sinaplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None,
             bw="scott", cut=2, scale="count", scale_hue=True, gridsize=100,
             violin=True, inner=None, 
             width=.8, split=False, dodge=True, orient=None,
             linewidth=1, color=None, palette=None, saturation=.75, violin_facealpha=0.25,
             point_linewidth=None, point_size=5, point_edgecolor="none", point_facealpha=1,
             legend=True, random_state=None, ax=None, **kwargs):

    plotter = _SinaPlotter(x, y, hue, data, order, hue_order,
                           bw, cut, scale, scale_hue, gridsize,
                           width, inner, split, dodge, orient, linewidth,
                           color, palette, saturation,
                           violin_facealpha, point_facealpha)

    np.random.seed(random_state)
    point_size = kwargs.get("s", point_size)
    if point_linewidth is None:
        point_linewidth = point_size / 10
    if point_edgecolor == "gray":
        point_edgecolor = plotter.gray
    kwargs.update(dict(s=point_size ** 2,
                       edgecolor=point_edgecolor,
                       linewidth=point_linewidth,
                       violin=violin))

    if ax is None:
        ax = plt.gca()

    plotter.plot(ax, kwargs)
    if not legend:
        ax.legend_.remove()
    return ax



def pheno_plot(data, plot_vars, cmap, sample, outpath=None):
    
    from sklearn.cluster import AgglomerativeClustering
    pheno = data[plot_vars]
    cluster = AgglomerativeClustering(affinity='euclidean', linkage='ward')
    cluster.fit_predict(pheno.T)
    order = pd.concat([pd.DataFrame(cluster.labels_), pd.DataFrame(pheno.columns)], axis = 1)
    order.columns = ['order', 'var']
    order = order.sort_values(by = 'order')
    var_order = list(order['var'])
    
    for col in pheno.columns:
        pheno[col] = (pheno[col] - pheno[col].mean())/pheno[col].std(ddof=0)
              
    columns = pheno.columns
    
    columns = columns.str.replace('strp_scr_acc_', '', regex=True)
    columns = columns.str.replace('strp_scr_', '', regex=True)
    columns = columns.str.replace('_', ' ', regex=True) 
    
    columns = columns.str.replace('new', '', regex=True)
    columns = columns.str.replace('Acc', '', regex=True)
    #columns = columns.str.replace('acc', '', regex=True)
    columns = columns.str.replace('all', '', regex=True)
    columns = columns.str.replace('Eq', '', regex=True)
    columns = columns.str.replace('MC', '', regex=True)
    columns = columns.str.replace('Minus', '-', regex=True)

    
    pheno.columns = columns 
    cluster = data.filter(regex='Subtype')
    pheno = pd.concat([cluster, pheno], axis=1)
    pheno_long = pd.melt(pheno, id_vars=['Subtype'], value_vars=columns)    
    plt.style.use('fivethirtyeight') 
    plt.figure(figsize=(14,7))
    ax = sns.lineplot(data=pheno_long, x="variable", y="value",
                      hue="Subtype", palette=cmap, sort= False)

    ax.grid(axis='x')
    plt.xticks(rotation = 45, ha = 'right', rotation_mode='anchor', size = 20)
    plt.yticks(size = 20)
    plt.legend(title = "Subtype", loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set(xlabel = sample)
    #plt.ylim((-1.5, 1.5)) 
    #plt.title(brain_region.capitalize())
    plt.tight_layout()
    ax.spines['bottom'].set_color('black')
    ax.xaxis.set_ticks_position('bottom') 
    ax.set_ylabel("Z-Score Residuals")
    

def barplot(data, x, y, cmap, xlab, ylab, title):

    from matplotlib import pyplot as plt
    import seaborn as sns
    #%matplotlib inline
    sns.set(font='Arial')
    plt.rcParams['svg.fonttype'] = 'none'
    style = sns.axes_style('white')
    style.update(sns.axes_style('ticks'))
    style['xtick.major.size'] = .5
    style['ytick.major.size'] = .5
    sns.set(font_scale=2)
    plt.style.use('fivethirtyeight')
    
    def change_width(ax, new_value) :
        for patch in ax.patches :
            current_width = patch.get_width()
            diff = current_width - new_value

            # we change the bar width
            patch.set_width(new_value)

            # we recenter the bar
            patch.set_x(patch.get_x() + diff * .5)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax = sns.barplot(data=data, x=x, y=y, palette=cmap, dodge=False)
    plt.setp(ax.artists,fill=False, color = '#94041a') 
    ax.grid(axis='x')
    ax.set(ylabel = ylab)
    ax.set(xlabel = xlab)
    ax.set_yticklabels(ax.get_yticks(), size = 20)
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    plt.title(title)
    #plt.ylim((.92, .96)) 
    #ax.set_yticklabels([0,1,2,3,4,5,6,""])
    plt.tight_layout()
    ax.spines['bottom'].set_color('black')
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.grid(False)
    change_width(ax, .7)
    ax.set_box_aspect(5/len(ax.patches)) 
    #plt.show()



def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.0f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center", size = 18) 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
        
def plot_sig_comps(data):
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x="pairs", y="count", data=data)
    show_values_on_bars(ax)
    plt.setp(ax.artists,fill=False, color = '#94041a') 
    ax.grid(axis='x')
    ax.set(ylabel = "Count")
    ax.set(xlabel = "Pairwise Comparisons")
    ax.set_yticklabels(ax.get_yticks(), size = 20)
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    plt.title('Number of significant pairwise comparisons')
    plt.tight_layout()
    ax.spines['bottom'].set_color('black')
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.grid(False)
    plt.show()
    
    
def cor_mat(data, cols):
    from string import ascii_letters
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="white")

    # Generate a large random dataset
    rs = np.random.RandomState(33)
    d = pd.DataFrame(data=rs.normal(size=(100, 26)),
                     columns=list(ascii_letters[26:]))

    # Compute the correlation matrix
    corr = data[cols].corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    corplot = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    return corplot


def getF(data, var, formula):
    data[var] = data[var].astype('float')
    model = ols(var + formula, data=data).fit()
    anova_table = np.array(sm.stats.anova_lm(model, type=2)[['F', 'PR(>F)']])[0]
    return anova_table


def getposthoc(data, var, formula):
    data[var] = data[var].astype('float')
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

def run_sample_anovas(data, anova_vars, formula):
    
    global df1_rest_idsub, df2_rest_idsub, full_sample_idsub, new_demos, study_site
  
    from timeit import default_timer as timer
    start = timer()
    
    sample_tests = []
    for i,j in zip([df1_rest_idsub, df2_rest_idsub, full_sample_idsub], ['Sample1', 'Sample2', 'Full_Sample']):
        test = pd.merge(i, data, on = 'ID')
        test = pd.merge(test, new_demos[['ID', 'age','sex_at_birth']], on ='ID')
        test = pd.merge(test, study_site.drop_duplicates('ID'), on ='ID')
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


import numpy as np
from statsmodels.stats.proportion import proportions_ztest

def prop_test(data, Sub_Group, Var, Col):
    
    g1 = data[data[Var] == Col].groupby(Sub_Group).count()[Var].loc[1]
    g2 = data[data[Var] == Col].groupby(Sub_Group).count()[Var].loc[2]
    g3 = data[data[Var] == Col].groupby(Sub_Group).count()[Var].loc[3]
    g4 = data[data[Var] == Col].groupby(Sub_Group).count()[Var].loc[4]
    
    onevtwo = proportions_ztest([g1, g2], [len(data[data[Sub_Group] == 1]), len(data[data[Sub_Group] == 2])])
    onevthree = proportions_ztest([g1, g3], [len(data[data[Sub_Group] == 1]), len(data[data[Sub_Group] == 3])])
    onevfour = proportions_ztest([g1, g4], [len(data[data[Sub_Group] == 1]), len(data[data[Sub_Group] == 4])])
   
    twovthree = proportions_ztest([g2, g3], [len(data[data[Sub_Group] == 2]), len(data[data[Sub_Group] == 3])])
    twovfour = proportions_ztest([g2, g3], [len(data[data[Sub_Group] == 2]), len(data[data[Sub_Group] == 4])])
    
    threevfour = proportions_ztest([g3, g4], [len(data[data[Sub_Group] == 3]), len(data[data[Sub_Group] == 4])])
    
    out = [onevtwo, onevthree, twovthree,  threevfour]
    return out

def get_props(overlap):

    props=[]
    for i in range(1,5):
        props.append(
            (pd.DataFrame(prop_test(overlap, 'Subtype_x', 'Subtype_y',  i), columns = ['F', 'p'])
             .assign(Group = i))[['Group', 'p']].T
        )
    output =  (pd.concat(props)
               .round(3)
               .reset_index()
               .query('index != "Group"')
               .drop('index', axis=1)
               .reset_index(drop=True)
               #.assign(Group = list(range(1,5)))
    )
    
    return output
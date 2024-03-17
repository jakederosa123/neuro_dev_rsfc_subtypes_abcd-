set.seed(111)
#if (!require(tidyverse)) {install.packages("tidyverse")}
#if (!require(readxl)) {install.packages("readxl")}
#if (!require(randomForest)) {install.packages("randomForest")}
#if (!require(sjstats)) {install.packages("sjstats")}
#if (!require(doParallel)) {install.packages("doParallel")}
#if (!require(party)) {install.packages("party")}
#if (!require(data.table)) {install.packages("data.table")}
#if (!require(esc)) {install.packages("esc")}
#if (!require(esc)) {install.packages("mltools")}

#urlPackage <- "https://cran.r-project.org/src/contrib/Archive/randomForest/randomForest_4.6-14.tar.gz"
#install.packages(urlPackage, repos=NULL, type="source") 
#install.packages("permimp", repos = "https://cloud.r-project.org/")

library(tidyverse)
library(permimp)
#library(sjstats)
library(fastDummies)
#library(doParallel)
#library(randomForest)
library(ggplot2)
library(esc)
library(mltools)
library(data.table)
library(party)

sample1_all = read.csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/prediction/sample1_all.csv') %>%
    select(ID, Subtype, everything()) %>%
    mutate(Subtype = as.factor(Subtype)) 

sample2_all = read.csv('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/prediction/sample2_all.csv') %>%
    select(ID, Subtype, everything()) %>%
    mutate(Subtype = as.factor(Subtype)) 


# Store the directory path
dir_path = '/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/prediction/varimp_boruta_fts'

# List the CSV files in the directory with full path
csv_files_full_path = list.files(path = dir_path, pattern = "\\.csv$", full.names = TRUE)
# Get the file names without the .csv extension
pred_names = sub("\\.csv$", "", basename(csv_files_full_path))

# Read in the CSV files and store them in a list
data_list <- lapply(csv_files_full_path, function(file) {
  read.csv(file, stringsAsFactors = FALSE)
})

# Name the list items based on pred_names
named_data_list <- setNames(data_list, pred_names)

run_forest = function(data, predictor, cols, sample, itr){
    
  num_tree = 100

  #run_df = data %>% 
  #  mutate(rf_predictor = UQ(sym(predictor))) %>%
  #  select(rf_predictor, Filter(function(x) x != predictor, cols)) %>%
  #  drop_na() 
  
  # party_domain <- cforest(
  #  formula = rf_predictor ~ .,
  #  data = run_df,
  #  control = cforest_unbiased(
      #mtry = ceiling(sqrt(length(cols))),
  #    mtry = round((ncol(run_df) - 4)/3),
  #    ntree = num_tree,   
  # ))
    
  #VIM_temp = permimp(party_domain, conditional = FALSE, asParty = TRUE, threshold = .99)
  #VIM_tempdf = as.data.frame(VIM_temp$values) %>% rename(VIM_temp = c(1))
  
  #new_cols = VIM_tempdf %>%
  #  arrange(desc(VIM_temp)) %>%
  #  mutate(VIM_temp = VIM_temp*10000000) %>%
  #  slice(c(1:15)) %>% rownames()
  
  subtype_cols = c(c('Subtype'), cols)
  #subtype_cols = c(c('Subtype'), cols)

  run_df = data %>% 
    mutate(rf_predictor = UQ(sym(predictor))) %>%
    select(rf_predictor, Filter(function(x) x != predictor, subtype_cols)) %>%
    drop_na() %>%
    mutate(Subtype = as.factor(Subtype))

  
  party_domain <- cforest(
    formula = rf_predictor ~ .,
    data = run_df,
    control = cforest_unbiased(
      #mtry = ceiling(sqrt(length(cols))),
      mtry = round((ncol(run_df) - 4)/3),
      ntree = num_tree,
      
    ))

  #assign("party_domain",party_domain,envir = .GlobalEnv)
  
  predictions <-predict(party_domain)
  party_domain_r=cor(predictions,run_df$rf_predictor)
  print(paste0(predictor, " r: ", round(party_domain_r[1],3)))
  
  party_domain_r_squared=cor(predictions,run_df$rf_predictor)**2
  print(paste0(predictor, " r-squared: ", round(party_domain_r_squared[1],3)))

  VIM_temp = permimp(party_domain, conditional = FALSE, asParty = TRUE, threshold = .99)
  #VIM_temp = varimp(party_domain, conditional = FALSE)

  print(VIM_temp)
  VIM_tempdf = as.data.frame(VIM_temp$values) %>% rename(VIM_temp = c(1))
  
  VIM_tempdf_2 = VIM_tempdf %>%
    mutate(Variable = rownames(.),
           r =  party_domain_r, 
           r2 = party_domain_r_squared) %>% 
    #filter(VIM_temp > 0) %>%
    arrange(desc(VIM_temp)) %>%
    mutate(Vim_temp_ranked = max(rank(VIM_temp)) + 1 - rank(VIM_temp)) 
  
  write.csv(VIM_tempdf_2, paste0('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/prediction/varimp_output/', predictor, '_varimp_', sample, '_', itr, '_permimp.csv'))
 
  #plt = ggplot(data = VIM_tempdf_2) +
  #       geom_bar(stat= "identity", aes(x =reorder(Variable, VIM_temp), y = VIM_temp), fill = '#C70039') +
  #ylim(0,0.01) +
  #       coord_flip() +
  #       theme_linedraw() +
  #       ggtitle(paste0(toupper(predictor))) +
  #theme(
  #  axis.title.y = element_blank(),
  #  plot.subtitle = element_text(size=22, hjust = .5),
  #  title = element_text(hjust = .5, size = 35),
  #  axis.title.x = element_text(size=35),
  #  axis.text = element_text(size = 25),
  #  plot.margin = unit(c(2,2,2,2), "cm")
  #) +
  #ylab("Decrease in Accuracy")
                                
  #ggsave(plt, height = 20, width = 30, 
  #       file = paste0('/pl/active/banich/studies/Relevantstudies/abcd/data/clustering/analysis/prediction/varimp_output/', predictor, '_varimp_', sample, 'permimp.png'))

  return(VIM_tempdf_2)
}
                                

#pred_names = c('CommonEF', 'Intelligence', 'UpdatingSpecific', 'pc1_new_r', 'pc2_new_r', 'pc3_new_r')
                                
#pred_names = c('withdrawn_depressed_r','somatic_complaints_r','social_problems_r',
#               'thought_problems_r','attention_problems_r','rule_breaking_r','agressive_r', 
#               'externalizing_r', 'internalizing_r', 'total_r', 
#               'predmeditation', 'perserverance', 'sensation_seeking','negative_urgency', 'positive_urgency',
#               'Stroop_interf_acc_all_r','Happy_Acc_Eq_r','Angry_Acc_Eq_r', 'LMT_r', 'RAVLT_r', 'perserverance)
     
                                
for(s in seq(510, 1000)){
    for(i in pred_names){
       #print(i)
       cols = c(named_data_list[[i]]$features)
       run_forest(sample1_all, i, cols, 'sample1', s)
       run_forest(sample2_all, i, cols, 'sample2', s)
    }
}
                                


import pandas as pd
import os
import stanza
#from tqdm import tqdm
from tqdm.auto import tqdm
import time
#import pickle
import numpy as np
import math
from methods import *



#### VARIABLES ####
#input_path = input('Enter file path: ')
#output_name = input('Enter output file name ')

#input_path = 'dataset/Longitudinal'
#output_path = 'output_long.csv'
#existing_df = 'output.csv'

#### SETTINGS ####
step_1_bool = True
remove_pauses = True
remove_null_utt = True
age_cat_bool = True
run_stanza_bool = True
existing_df_bool = False
get_tag_info_bool = True

def run_main(input_path,output_path):
    #### PROCESS ####

    if(existing_df_bool): df = pd.read_csv(existing_df)

    #1 PROCESS FOLDERS
    df = pd.DataFrame(columns=['utterance','speaker','child','age','PID','date','file'])
    df = process_folders(input_path,df)

    print('Files import complete')


    #2 REMOVE NULL UTTERANCES

    df = df[df['utterance']!='0 .']
    df = df.reset_index(drop=True)
    print('Null utterances removed')



    #3 CREATE AGE BINS

    df['age'] = df['age']*12
    bins = np.arange(math.floor(df['age'].min()), math.ceil(df['age'].max())+3,3)
    df['age_cat'] = pd.cut(df.age, bins = bins, right=False)

    print('Age bins complete')



    #4 REMOVE PAUSES


    df['pause_count'] = df['utterance'].apply(check_pause)
    max_pause = df.pause_count.max()
    for i in range(max_pause):
        for index, row in (df[df.pause_count>0]).iterrows():
            split_line(df,index)
        df = df.sort_index().reset_index(drop=True)
    print('Remove pauses complete.')




    #5 RUN STANZA
    nlp = stanza.Pipeline(lang='it', processors='tokenize,mwt,pos,lemma,depparse')

    def get_tags(s):
        return nlp(s)

    tqdm.pandas()

    df['tags'] = df['utterance'].progress_apply(get_tags)
    print('Stanza process complete')
    #df.to_csv(output_name)



    #6 GET TAGS

    df['tag_df'] = df['tags'].apply(tag_df)

    df['nsubj'] = df['tag_df'].apply(get_nsubj)
    df['imp_count'] = df['tag_df'].apply(imp_count)
    df['imp'] = df['tag_df'].apply(get_imp)
    df['pas_count'] = df['tag_df'].apply(pas_count)
    df['pas'] = df['tag_df'].apply(get_pas)
    df['fut_count'] = df['tag_df'].apply(fut_count)
    df['fut'] = df['tag_df'].apply(get_fut)
    df['pres_count'] = df['tag_df'].apply(pres_count)
    df['pres'] = df['tag_df'].apply(get_pres)

    df['pron_count'] = df['tag_df'].apply(pron_count)
    df['pron'] = df['tag_df'].apply(get_pron)

    df['propn_count'] = df['tag_df'].apply(propn_count)
    df['prop'] = df['tag_df'].apply(get_propn)

    df['noun_count'] = df['tag_df'].apply(noun_count)
    df['noun'] = df['tag_df'].apply(get_noun)

    print('Done')




    # OUTPUT

    #check file name extension
    #output_name = output_name.split('.')[0]+'.csv'
    df.to_csv(output_path)
    print('Saved.')

run_main('dataset/Longitudinal','output_long.csv')
run_main('dataset/Cross','output_cross.csv')

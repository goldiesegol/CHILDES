import pandas as pd
import os
import stanza
import json
from stanza.models.common.doc import Document

def rename(folder):
    path = folder +"/"
    for filename in os.listdir(folder):
        os.rename(path+filename, path+filename[:(len(filename)-3)]+"txt")

def get_age(s):
    y = int(s.split(';')[0])
    m = int(s.split(';')[1].split('.')[0])
    d = int(s.split(';')[1].split('.')[1])
    age_ = y + (m/12) + (d/365)

    return round(age_,2)

def process_folder(folder,df):
    rename(folder)
    path = folder +"/"

    result = list()

    for file in os.listdir(folder):
        if('.' != file[0]):
            f = path+file
            result.append(process_file(f))
    return pd.concat(result,ignore_index=True)

def process_file(file):
    foo = open(file)
    f = foo.readlines()
    print(file)

    newline=False
    meta = list()
    words = pd.DataFrame(columns = ['utterance','speaker',])
    morphology = ''
    grammar = ''

    for line in f:
        marker = line[0]
        if(marker == '@'): meta.append(line)

        elif(marker == '*'):

            if(newline):
                words.loc[len(words.index)] = [utt, speaker]

            utt = (line.split(':')[1]).strip()
            speaker = line.split(':')[0]
            newline = True

    words.loc[len(words.index)] = [utt, speaker]

    #now process the meta data
    for elem in meta:
        if ':' in elem:
            m = elem.split(':')[0]
            d = elem.split(':')[1]

            if(m=='@Participants'):
                child = d.split()[1]
            elif(m=='@PID'):
                PID = d
            elif(m=='@ID'):
                if(file == 'dataset/Longitudinal/Diana/020613.txt'):
                    age = 2.54
                elif(file=='dataset/Longitudinal/Francesco/010707.txt'):
                    age = 1.6
                else:
                    numer = False
                    for char in d:
                        if(char.isdigit()): numer = True
                    if(numer): age = get_age(d.split('|')[3])
            elif(m=='@Date'):
                date = d
    words['child'] = child
    words['age'] = age
    words['PID'] = PID.strip()
    words['date'] = date.strip()
    words['file'] = file.split('/')[-1]

    return words

def process_folders(path,df):
    result = list()
    for folder in os.listdir(path):
        f_path = (path+'/'+folder)
        if('.' not in folder):
            result.append(process_folder(f_path,df))
    return pd.concat(result,ignore_index=True)

def check_pause(s):
    count = 0
    for item in s.split():
        if (item == """(.)"""): count = count + 1
    return count

def get_pause_indexes(df):
    result = []
    for index,row in df.iterrows():
        if check_pause(row.utterance)>0:
            result.append(index)
    return result

def split_line(df, index):
    row = df.iloc[index]
    ut = row.utterance.split()
    id = -1

    for i in range(len(ut)):
        if ut[i] == """(.)""":
            id = i
    if(id>=0):
        line1 = row.copy()
        line2 = row.copy()

        line1.utterance = ' '.join(ut[:id])
        line2.utterance = ' '.join(ut[id+1:])

        df.loc[index] = line1
        df.loc[index+.5] = line2

def get_feat_1(s,feat):
    result = ''
    if(s is not None):
        foo = s.split('|')
        for item in foo:
            if item.split('=')[0]==feat: result = item.split('=')[1]
    return result

def tag_df(tags):
    result = pd.DataFrame(columns=['text', 'id','lemma','upos','xpos','head','deprel','Tense','Definite','Gender','Number','PronType','Mood','VerbForm','Aspect'])
    if(type(tags) is str):
        tags = Document(json.loads(tags))
    for sent in tags.sentences:
        for word in sent.words:
            Definite = get_feat_1(word.feats,'Definite')
            Gender = get_feat_1(word.feats,'Gender')
            Number = get_feat_1(word.feats,'Number')
            PronType = get_feat_1(word.feats,'PronType')
            Mood = get_feat_1(word.feats,'Mood')
            Tense = get_feat_1(word.feats,'Tense')
            VerbForm = get_feat_1(word.feats,'VerbForm')
            Aspect = get_feat_1(word.feats,'Aspect')
            result.loc[len(result.index)] = [word.text,word.id,word.lemma,word.upos,word.xpos,word.head,word.deprel,Tense,Definite,Gender,Number,PronType,Mood,VerbForm,Aspect]
    return result

def get_feat(df,m,v):
    result = []
    if m in df.columns:
        for index,row in df.iterrows():
            if(row[m]==v):
                result.append(row.text)
    return result

def contains_feat_count(df,m,v):
    count = 0
    if m in df.columns:
        for index,row in df.iterrows():
            if(row[m]==v):
                count = count+1
    return count

def get_nsubj(df):
    return(get_feat(df,'deprel','nsubj'))

def get_final_punct(s):
    if(len(s)>0):
        return s.split()[-1]
    else: return ''

def full_sentence(p):
    p_set = ['.','!','?']
    if p in p_set:
        return True
    else: return False

def imp_count(df):
    return contains_feat_count(df,'Tense','Imp')

def get_imp(df):
    return(get_feat(df,'Tense','Imp'))

def pas_count(df):
    return contains_feat_count(df,'Tense','Past')

def get_pas(df):
    return(get_feat(df,'Tense','Past'))


def fut_count(df):
    return contains_feat_count(df,'Tense','Fut')

def get_fut(df):
    return(get_feat(df,'Tense','Fut'))

def pres_count(df):
    return contains_feat_count(df,'Tense','Pres')

def get_pres(df):
    return(get_feat(df,'Tense','Pres'))

def pron_count(df):
    return(contains_feat_count(df,'upos','PRON'))

def get_pron(df):
    return(get_feat(df,'upos','PRON'))

def propn_count(df):
    return(contains_feat_count(df,'upos','PROPN'))

def get_propn(df):
    return(get_feat(df,'upos','PROPN'))

def noun_count(df):
    return(contains_feat_count(df,'upos','NOUN'))

def get_noun(df):
    return(get_feat(df,'upos','NOUN'))

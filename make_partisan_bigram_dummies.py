'''
    make_partisan_bigram_dummies.py
    
    > Reads in partisanship scores of different congressional bigrams from Gentzkow et al. (2020) and identifies the 
    1000 most Republican and Democrat-leaning bigrams over the time-span of the sample.
    
    > Generates speech-level dummies for each of these bigrams which will later be used in Stata to estimate the 
    probabilities (and their standard errors) of the bigrams being spoken in speeches where God has and has not been 
    evoked.
    
    > Each party's partisan bigram dummies stored in a .csv file: "data/<rep/dem>_partisan_bigram_dummies.csv"

'''

import time as tim
import pandas as pd

## Create mapping from Congressional session to year (as bigram partisanship scores are by session) ##

start_year = 1949
end_year = 2014
start_session = 81

congress_sessions = {}
session = start_session
for y in range(end_year-start_year+1):
    year = start_year+y
    
    congress_sessions[year] = session

    if y % 2:
        session = session+1


# Read in corpus dictionary (gensim dictionary object) ##

dct = pd.read_pickle('data/dct.pkl')


# Identify 1000 most republican and democrat-leaning bigrams ##

# retrieve bigrams and their partisanship scores from text files from all sessions
partisan_bigrams_scores_all = []
for session in range(congress_sessions[start_year],congress_sessions[end_year]+1):       
    
    session_file = f'partisan_phrases_0{session}.txt' if session < 100 else f'partisan_phrases_{session}.txt'        
    with open(f'data/gentzkow/phrase_partisanship/{session_file}', 'r', encoding='utf-8') as f:
        session_partisan_bigrams_rows = f.readlines()[1:]                
    
    session_partisan_bigrams_scores  = [[row[:row.index('|')].replace(' ', '_').strip(), float(row[row.index('|')+1:])] 
                                        for row in session_partisan_bigrams_rows]
    
    partisan_bigrams_scores_all += session_partisan_bigrams_scores

# for each unique partisan bigram, sum across all sessions to generate its total score over the sample
partisan_bigrams = list({bigram for bigram, score in partisan_bigrams_scores_all})
partisan_bigrams_scores = []
for partisan_bigram in partisan_bigrams:
    partisan_bigram_score = sum([score for bigram, score in partisan_bigrams_scores_all if bigram==partisan_bigram])
    partisan_bigrams_scores.append([partisan_bigram, partisan_bigram_score])

# identify the 1000 most republican and democrat-leaning bigrams (republican: >0 ; democrat <0) (See Section 3.1.1)
df = pd.DataFrame(partisan_bigrams_scores, 
                  columns=['partisan_bigram','score']).sort_values(by='score').reset_index(drop=True)
df_rep = df[df.score>0].sort_values(by='score',ascending=False).reset_index(drop=True)
df_dem = df[df.score<0].sort_values(by='score').reset_index(drop=True)

df_rep['rank'] = [index+1 for index in df_rep.index.to_list()]
df_dem['rank'] = [index+1 for index in df_dem.index.to_list()]

rep_partisan_bigrams = df_rep.loc[:999,'partisan_bigram'].tolist()
dem_partisan_bigrams = df_dem.loc[:999,'partisan_bigram'].tolist()
pd.to_pickle(rep_partisan_bigrams, 'data/rep_partisan_bigrams.pkl')
pd.to_pickle(dem_partisan_bigrams, 'data/dem_partisan_bigrams.pkl')

rep_partisan_bigram_ids = [dct.token2id[token] for token in rep_partisan_bigrams]
dem_partisan_bigram_ids = [dct.token2id[token] for token in dem_partisan_bigrams]


## Generate speech-level dummies for each partisan bigram ##

start_year = 1950
end_year = 2014
t = tim.time()

for chamber in ['H','S']:

    for year in range(start_year, end_year+1):
    
        # tracks running time
        secs = tim.time() - t
        days, secs = secs // 86400, secs % 86400
        hours, secs = secs // 3600, secs % 3600
        mins, secs = secs // 60, secs % 60 
        print(f'Elapsed time: {int(days)}d:{int(hours)}h:{int(mins):0>2}m:{int(secs):0>2}s')
    
        # determine index of Republican / Democrat speeches using the speech-level metadata
        year_metadata = pd.read_pickle(f'data/congress/{chamber}/{year}-metadata.pkl')        
        year_rep_index = year_metadata[year_metadata.republican==1].index.tolist()
        year_dem_index = year_metadata[year_metadata.democrat==1].index.tolist()   
        
        # read in speech-level god dummies (so can calculate probabilities for speeches which do / do not evoke God)
        year_god_dummies = pd.read_pickle(f'data/congress/{chamber}/{year}-god_dummies.pkl')       
        year_rep_god_dummies = [year_god_dummies[i] for i in year_rep_index]
        year_dem_god_dummies = [year_god_dummies[i] for i in year_dem_index]
           
        # read in speech-level term frequencies        
        year_tfs = pd.read_pickle(f'data/congress/{chamber}/{year}-tfs.pkl')      
        year_rep_tfs = [year_tfs[i] for i in year_rep_index]
        year_dem_tfs = [year_tfs[i] for i in year_dem_index]
        
        # generate speech-level partisan dummies using the term frequencies
        year_rep_partisan_dummies = [[1 if partisan_bigram_id in [token_id for token_id, tf in speech_tfs] else 0 
                                      for partisan_bigram_id in rep_partisan_bigram_ids] for speech_tfs in year_rep_tfs]
        
        year_dem_partisan_dummies = [[1 if partisan_bigram_id in [token_id for token_id, tf in speech_tfs] else 0 
                                      for partisan_bigram_id in dem_partisan_bigram_ids] for speech_tfs in year_dem_tfs]
        
        year_rep_partisan_dummies = pd.DataFrame(year_rep_partisan_dummies,columns=rep_partisan_bigrams)        
        year_dem_partisan_dummies = pd.DataFrame(year_dem_partisan_dummies,columns=dem_partisan_bigrams)
        
        # create column with speech-level god dummy variable
        year_rep_partisan_dummies['god'] = year_rep_god_dummies
        year_dem_partisan_dummies['god'] = year_dem_god_dummies
       
        if chamber=='H' and year==start_year:
            year_rep_partisan_dummies.to_csv('data/rep_partisan_bigram_dummies.csv', index=False)
            year_dem_partisan_dummies.to_csv('data/dem_partisan_bigram_dummies.csv', index=False)
        else:
            year_rep_partisan_dummies.to_csv('data/rep_partisan_bigram_dummies.csv',mode='a', index=False, header=False)
            year_dem_partisan_dummies.to_csv('data/dem_partisan_bigram_dummies.csv',mode='a', index=False, header=False)
            
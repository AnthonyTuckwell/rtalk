'''
    make_speech_partisanship.py
    
    >  Reads in the speech-level metadata, god dummies and partisanship scores and combines them into a Stata .dta file 
    (data/speech_data.dta)

'''


import pandas as pd
import time as tim
import math

## Create mapping from Congressional session to year ##

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
        

## Read in speech-level metadata, god dummies and partisanship scores and combine into Stata .dta file ##

speech_data = pd.DataFrame()

start_year = 1950
end_year = 2014
t = tim.time()

for chamber in ['H','S']:

    for year in range(start_year, end_year+1):
        
        print(f'\n\nChamber: {chamber}. Year: {year}')
        secs = tim.time() - t
        days, secs = secs // 86400, secs % 86400
        hours, secs = secs // 3600, secs % 3600
        mins, secs = secs // 60, secs % 60 
        print(f'Elapsed time: {int(days)}d:{int(hours)}h:{int(mins):0>2}m:{int(secs):0>2}s')        
        
        year_metadata = pd.read_pickle(f'data/congress/{chamber}/{year}-metadata.pkl')
        year_god_dummies = pd.read_pickle(f'data/congress/{chamber}/{year}-god_dummies.pkl')
        year_bias = pd.read_pickle(f'data/congress/{chamber}/{year}-partisan_scores.pkl')
                
        year_data = year_metadata.copy()
        year_data['god_dummy'] = year_god_dummies
        year_data['bias'] = year_bias
    
        speech_data = speech_data.append(year_data)
        

speech_data['session'] = speech_data['year'].apply(lambda x: congress_sessions[x])
speech_data['decade'] = speech_data['year'].apply(lambda x: math.floor(int(str(x))/10)*10)

obj_type_cols = list(speech_data.select_dtypes(include=['object']).columns)
speech_data[obj_type_cols] = speech_data[obj_type_cols].astype(str)

speech_data.to_stata('data/speech_data.dta', write_index=False, version=117)


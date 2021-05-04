'''
    make_term_frequencies.py
    
    > Reads in congressional speeches' text data and metadata 
    
    > Creates the following speech-level data:
        > term frequencies (counts of bigrams);
        > god dummies; 
        > speech metadata
        
    > Variables stored by year in "data/congress"

'''

from datetime import timedelta, date, time
import time as tim
import pandas as pd
import os, re
import gensim
import Stemmer
import nltk


## Define iterators / functions ##

# date iterator
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

# cleans congressional speech
def clean_text(text):
    text = re.sub('-\n', '', text) 
    text = text.replace('\n', ' ')
    text = text.replace(u'\xa0', u' ')
    text = re.sub(' +', ' ', text)
    return text


## Make stopword list ## 

# read in snowball stopwords
with open('data/snowball_stopwords.txt', 'r') as f:
    stopwords_rows = f.readlines()
stopwords = [re.findall(r"^[a-z']+", row)[0] for row in stopwords_rows if 
             len(re.findall(r"^[a-z']+", row))>0]

# expand contractions ("i'll", etc.) into component parts (e.g. "i", "ll"), and add to list
contractions = [word.split("'") for word in stopwords if "'" in word]
contractions = [word for contraction in contractions for word in contraction]
stopwords = set([word for word in stopwords if "'" not in word] + contractions)


## Make dictionary for tf counts ## 

# create corpus dictionary for tf counts using same vocab as gentzkow (to align preprocessing steps)
with open('data/gentzkow/vocab.txt','r',encoding='utf-8') as f:
    vocab_rows = f.readlines()
    vocab = [row.strip().replace(' ', '_') for row in vocab_rows]
dct = gensim.corpora.Dictionary([vocab])
pd.to_pickle(dct,'data/dct.pkl')


## Initialise Porter2 stemmer ##

stemmer = Stemmer.Stemmer('english')


## Read in congress metadata ##

corpus_metadata = pd.read_stata('F:/corpora/congress/metadata_by_month.dta')


## Create speech-level metadata, term_frequencies and god_dummies (saved in lists for each year) ##

start_year = 1950
end_year = 2014
t = tim.time()

for chamber in ['H','S']:
    
    # create dummies to be used when identifying speech metadata
    house = 1 if chamber=='H' else 0
    senate = 1 if chamber=='S' else 0
        
    for year in range(start_year, end_year+1):
        
        print(f'\n\nChamber: {chamber}. Year: {year}')
        
        # tracks running time
        secs = tim.time() - t
        days, secs = secs // 86400, secs % 86400
        hours, secs = secs // 3600, secs % 3600
        mins, secs = secs // 60, secs % 60 
        print(f'Elapsed time: {int(days)}d:{int(hours)}h:{int(mins):0>2}m:{int(secs):0>2}s')
        
        # create empty lists to store speech-level data for the current year
        year_metadata = []
        year_tfs = []
        year_god_dummies = []
    
        for date_obj in daterange(date(year, 1, 1), date(year+1, 1, 1)):
        
            date_str = f'{date_obj.strftime("%Y")}-{date_obj.strftime("%m")}-{date_obj.strftime("%d")}'
            
            # use Government Publishing Office data from year its available (1994) ; previously, use HeinOnline data
            if date_obj.year>=1994:
                 dataset = 'gpo'
            else:
                dataset = 'hein'
                                 
            path = f'F:/corpora/congress/{dataset}/{chamber}/{date_str}'
                    
            # if no speeches found for the current date, continue
            try:
                speeches_filenames = os.listdir(f'{path}')
            except:
                continue 
            
            # retrieve congress metadata for the year and month corresponding to the current date
            month_metadata = corpus_metadata[(corpus_metadata['speech_year']==date_obj.year)
                                             &(corpus_metadata['speech_month']==date_obj.month)]

            for filename in speeches_filenames:
                                           
                # if filename not in <state>_<name>_<start> format continue (because not by a congress member)
                try:
                    state_name_start = re.findall(r'[HS]_([^_]+_[^_]+_[^_]+)', filename)[0]
                except:
                    continue  
                
                state = re.findall(r'([^_]+)_[^_]+_[^_]+', state_name_start)[0]
                name = re.findall(r'[^_]+_([^_]+)_[^_]+', state_name_start)[0].upper()
                start = re.findall(r'[^_]+_[^_]+_([^_]+)', state_name_start)[0].upper()               
            
                if state in ['CHAIRMAN', 'SPEAKER', 'NULL']:
                    continue

                # identify speaker metadata that correspnds to speech
                speaker_metadata = month_metadata[(month_metadata['state']==state) 
                                                & (month_metadata['last']==name)
                                                &(month_metadata['start']==int(start))
                                                &(month_metadata['house']==house)
                                                &(month_metadata['senate']==senate)]          
                
                # if no speaker metadata identified, continue
                if speaker_metadata.shape[0]:
                    speech_metadata = speaker_metadata.iloc[0].to_dict()                    
                else:
                    continue 
                
                speech_metadata = {'filename': filename, 'chamber': chamber, 'date': date_str, 
                                   'year': date_obj.year, 'month': date_obj.month, 'day': date_obj.day,
                                   **speech_metadata}     
                
                # read speech
                with open(f'{path}/{filename}', encoding='utf8') as f:
                    speech = f.read()
                
                # if speech empty, continue
                if speech.isspace():
                    continue
                
                # clean text
                speech = clean_text(speech)
                
                # coerce to lower-case
                speech = speech.lower()
                
                # create bigrams using same steps as Gentzkow et al. (2019)
                speech_words = [word for word in re.split('[^a-z0-9]+', speech) if word not in stopwords]
                speech_words_stem = [stemmer.stemWord(word) for word in speech_words]
                speech_bigrams_stem = ['_'.join(bigram) for bigram in nltk.ngrams(speech_words_stem,2)]
                
                # create term frequencies using vocab from Gentzkow et al. (2019) and god dummies
                speech_tfs = dct.doc2bow(speech_bigrams_stem)            
                god_dummy = 1 if 'god' in speech_words else 0
                
                year_tfs.append(speech_tfs)
                year_god_dummies.append(god_dummy)
                year_metadata.append(speech_metadata)
                     
        year_metadata = pd.DataFrame(year_metadata)
        
        pd.to_pickle(year_tfs, f'data/congress/{chamber}/{year}-tfs.pkl')
        pd.to_pickle(year_god_dummies, f'data/congress/{chamber}/{year}-god_dummies.pkl')
        year_metadata.to_pickle(f'data/congress/{chamber}/{year}-metadata.pkl')

            
        

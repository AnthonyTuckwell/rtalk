# rtalk

This repo provides code that accompanies the paper by Ash, Hills and Tuckwell, "The Political Economy of Religious Language: Evidence from the United States Congress", available here: 

The following scripts generate the data and results of the paper:

*make_term_frequencies.py*

This script reads in the corpus of congressional speeches and generates speech-level term frequencies (counts of bigrams), god dummies (whether the word "god" is in the speech) and metadata (including speaker characteristics, e.g. political party).

As mentioned in Section 3.2 of the paper, we align our pre-processing steps with Gentzkow et al. (2019) as our measure of speech partisanship makes use of partisanship scores of bigrams generated by the authors of that paper. Specifically: (1) The speech is coerced to lower-case. (2) The speech is broken into separate words, treating all non-alphanumeric characters as delimiters. (3) The same general English-language stopwords are removed  http://snowball.tartarus.org/algorithms/english/stop.txt). (4) Remaining words are reduced to stems using the Porter2 stemming algorithm. (5) The stemmed words are converted to bigrams following their order in the speech. Gentzkow et al. (2019) then further pre-process their bigrams, e.g. by removing bigrams which contain numbers, names of states / congresspeople, or which they identify as procedural. Provided as open-access are the final "valid" bigrams used in their analyss (see https://data.stanford.edu/congress_text). Our final step (6) is to filter the bigrams according to this vocabulary and count them. 
 

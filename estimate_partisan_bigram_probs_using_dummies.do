/*
	estimate_partisan_bigram_probs_using_dummies.do
	
	> Reads in speech-level dummies for the 1000 most Republican/Democrat-leaning bigrams generated 
	by "make_partisan_bigram_dummies.py"
	
	> Estimates the probabilities (as means of speech-level dummies) and standard errors of these bigrams being spoken
	 by Republicans/Democrats in speeches that do and do not evoke God.
	
	> Stores each party's probability data (partisan bigram probabilities (+ SEs) by God) in a 
	.csv file: "data/<rep/dem>_prob_data.csv"

*/


clear all
set more off
cls

cd "C:\Users\Anthony\Documents\Projects\religious-language-in-congress\data"


// Create probability data for Republican partisan bigrams //

import delimited "rep_partisan_bigram_dummies.csv"

ds god, not
local varlist `r(varlist)'
foreach V in `varlist' {
	 local meanlist `meanlist' m`V'=`V'
	 local selist `selist' se`V'=`V'
} 
collapse (mean) `meanlist' (semean) `selist', by(god)

save rep_prob_data.dta, replace

// Create probability data for Democrat partisan bigrams //

import delimited "dem_partisan_bigram_dummies.csv", clear

ds god, not
local varlist `r(varlist)'
foreach V in `varlist' {
	 local meanlist `meanlist' m`V'=`V'
	 local selist `selist' se`V'=`V'
} 
collapse (mean) `meanlist' (semean) `selist', by(god)

save dem_prob_data.dta, replace

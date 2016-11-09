# TriHiddenMarkovModel-POS
########################
Author: Dang Tran
Email: dangtm24@gmail.com
########################

This is a python code for TriHMM (Trigram Hidden Markov Model) using for POS (Part-of-speech) Tagging in NLP

FOR MORE INFORMATION ABOUT HOW TO WRITE THIS CODE BY YOURSELVES, PLEASE READ:

        http://www.cs.columbia.edu/~mcollins/courses/nlp2011/notes/hmms.pdf

(HOWEVER, I WILL TRY TO EXPLAIN TO YOU WHAT I AM WRITTING)

What is HMM ?

Hidden Markov Model is a Sequence Classifier (a classifier which depend on context of words in each sentences, 

ex: previous word, the word after, label of words around the one we considering,... )

Sequence Classifier opposite with Affix classifier, which only depend on morphology of each words (ex:

last character, first character, how many vowels in this word)

How HMM works?
I can explain this algorithm in very simple way:

1/ Because it is Sequence Classifier, it is prefer to working with a full sentence rather than few words
Which means if I classifier a whole sentence like this:

        "The man certainly didn't want to wait"

will work better than individual words like:
        
        "man" , "the" , "want"


2/ The Sequence Classifier will input a sentence x1x2x3...xn and then output POS label of each words y1y2y3...yn
EX:

        Input <- The man  certainly didn't want to wait

        Output-> Det NOUN ADVERB    VERB   VERB TO VERB

HERE IS HOW HMM ALGORITHM:
HMM try to combine all possibility of Tags 
        ex: if there are three main tags NOUN ADJ VERB, and the sentence have 4 character.It will try to combine all possibility : 
        
        NOUN NOUN NOUN NOUN,
        NOUN NOUN NOUN VERB,
        NOUN VERB NOUN NOUN,
        ...
        VERB VERB VERB VERB
Each of these combination will have a different "SCORE"
The algorithm will choose the best combination which has a "HIGHEST SCORE"


How to Compute "THIS SCORE" ?

If you want to know how to get this formula, please read:

        http://www.cs.columbia.edu/~mcollins/courses/nlp2011/notes/hmms.pdf

So, I only try to explain the code...

According to the formula, We need to compute two main probability  q(s|u,v) and e(x|s)

where q(s|u,v) :

        count(u,v,s)/count(u,v)

and e(x|s):

        count(x,s)/count(s)

I add two small Laplace smoothing for some reasons:

1/ Smoothing for q(s|u,v): We will try to combine all possibility. So it easily to have a combination that have count(u,v,s)=0

2/ Smoothing for e(x,s): x is a word, so probability that count(x,s)=0 will very high


Instead of trying to combine all possibility, we use Dynamic programming technique
For this problem, author propose Dynamic programming method called  Viterbi Algorithm

The rest of the code is interpreted from the paper.
import pandas as pd
import spacy
import math
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load("en")


def lemmatize(doc):
    """
Given an input spaCy document, it returns the lemmas of the words which are
not:
i)punctuations;
ii)white spaces;
iii)not stop words (except US); iv) "'s"; v)possesive pronouns (POS).

Parameters
----------
doc : spacy.tokens.doc.Doc
    document whose lemmas we want to find

Returns
-------
lemma: list
    lemmas of the input document in a list

"""
    lemma = [token.lemma_ for token in doc
             if not token.is_punct and not token.is_space
             and (token.text == "US"
                  or token.lower_ not in STOP_WORDS)
             and not token.tag_ == "POS" and not token.text == "'s"]
    return lemma


def tf(w, doc, normalize=False):
    """
It returns the term frequency of the chosen word (w) in the specified spaCy
document (doc).
Parameters
----------
w: string
    single word we want to compute the TF of
doc : spacy.tokens.doc.Doc
    document we want to compute the TF in
Returns
-------
message: string
    message to warn the user to choose a different w
tf_w: int, float
    number of times lemmatized w appears into lemmatized doc
"""
    # Taking the lemma of the input w
    w = lemmatize(nlp(w))
    # Checking whether the input word has a meaningful lemma
    if w == []:
        message = "The chosen word has not a meaningful lemma"
        return message
    else:
        # Taking the lemma of the input doc
        lem = lemmatize(doc)
        # Computing the tf
        tf_w = lem.count(w[0])  # I need to specify the position,
        # because w is a list.
        if normalize:
            tf_w = lem.count(w[0])/len(lem)
        return tf_w


def idf(w, doc_l, log_scale=False):
    """
It returns the inverse document frequency of the chosen word (w)
in the specified spaCy document (doc).

Parameters
----------
w: string
    single word we want to compute the IDF of
doc_l : list
    list of spaCy documents we want to comput the IDF in
log_scale: boolean
    Default is set to False. When True, it applies a
    logarithmic scale in the computation of the IDF.

Returns
-------
message: string
    message to warn the user to choose a different w
idf_w: int
    inverse document frequency.
    If log_scale = True, a logarithmic scale is applied to it.

"""
    # Taking the lemma of the input w
    w = lemmatize(nlp(w))
    # Checking whether the input word has a meaningful lemma
    if w == []:
        message = "The chosen word has not a meaningful lemma"
        return message
    else:
        # creating an empty list to store a placeholder
        # everytime the lemma of the word is in the list
        # of lemmas of the document
        idf_count = []
        for doc in doc_l:
            lem = lemmatize(doc)
            # checking whether the lemma of w appears
            # among the lemmas of the doc
            if w[0] in lem:
                idf_count.append("True")
        # Computing the tf
        idf_w = 1/(len(idf_count)+1)  # adding +1 to denominator
        # to avoid division by 0.
        if log_scale:
            # adding +1 to denominator to avoid division by 0.
            idf_w = math.log(len(doc_l)/(len(idf_count)+1))
        return idf_w


def tf_idf(w, doc, doc_l, scale=False):
    """
It returns the inverse document frequency of the chosen word (w) in the specified spaCy document (doc).

Parameters
----------
w: string
    single word we want to compute the IDF of
doc:spacy.tokens.doc.Doc
    document we want to compute the TF in
doc_l : list
    list of spaCy documents we want to comput the IDF in
scale: boolean
    Default is set to False. When True, it set normalize = True in the tf() function (it applies a Normalization)
    and log_scale = True in the idf function (it applies a logarithmic scale).

Returns
---------
message: string
    message to warn the user to choose a different w
tf_idf_w: float
    product of the tf() and idf() functions.

"""
    # calling the tf() function
    tf1 = tf(w, doc, normalize=scale)
    # calling the idf() function
    idf1 = idf(w, doc_l, log_scale=scale)
    # computing the product
    if type(tf1) != str and type(idf1) != str: #this is to prevent string type data
        #(coming from the non meaningful lemma messages) to be multiplied, since they cannot be.
        tf_idf_w = tf1*idf1
        return tf_idf_w
    else:
        message = "The chosen word has not a meaningful lemma"
        return message


def all_lemmas(doc_l):
    """
It returns the union set of all the lemmas in each document included in doc_l.

Parameters
----------

doc_l : list
    list of spaCy documents whose lemmas we want to find

Returns
-------
lemma_set: set
    set of all lemmas.
"""
    lemma_set = set()
    for doc in doc_l:
        lem = set(lemmatize(doc))
        lemma_set.update(lem)
    return lemma_set


def tf_idf_doc(doc, doc_l, scale=False):
    """
It returns a dictionary of {lemma: TF-IDF value}, corresponding to each the lemmas of all the available documents

Parameters
----------
doc: spacy.tokens.doc.Doc
    spaCy documents whose lemmas we want to compute the TF-IDF of
doc_l : list
    list of spaCy documents whose lemmas we want to comput the TF-IDF of
scale: boolean
    Default is set to False. When True, it uses the scaled version of the TF-IDF function

Returns
---------
tf_idf_dic: dict
    Dictionary with the TF-IDF of each lemmas in the chosen doc.
"""
    tf_idf_dic = {} #creating the empty dictionary
    for token in doc:
        tf_idf1 = tf_idf(token.text, doc,doc_l, scale)
        if type(tf_idf1) != str: #filtering out lemmas with non meaningful messages coming from the tf_idf() function
            tf_idf_dic[token.lemma_] = tf_idf1 #updating the dictionary
    return tf_idf_dic


def tf_idf_scores(doc_l, scale=False):
    """
It returns a pandas dataframe of the TF-IDF of the set of all lemmas of the chosen list of spaCy documens

Parameters
----------
doc_l : list
    list of spaCy documents whose lemmas we want to comput the TF-IDF of
scale: boolean
    Default is set to False. When True, it uses the scaled version of the TF-IDF function

Returns
---------
tfidf_df: pandas.core.frame.DataFrame
    Pandas DataFarme with the TF-IDF.
"""
    lemma_list = all_lemmas(doc_l)
    tfidf_dic = {} #the highest level dictionary, where the keys are the doc indexes
    #and the values are other dictionary
    for i, doc in enumerate(doc_l):
        tfidf_dic_sub = {} #the nested dictionary, where keys are lemmas, and values are the respective tfidf
        for l in lemma_list:
            tf_idf2 = tf_idf(l, doc, doc_l, scale)
            if type(tf_idf2) != str:
                tfidf_dic_sub[l] = tf_idf2 #updating the nested dictionary
        tfidf_dic["Doc "+str(i)] = tfidf_dic_sub #updating the high level dictionary

    tfidf_df = pd.DataFrame(tfidf_dic)
    tfidf_df = tfidf_df.transpose()
    return tfidf_df

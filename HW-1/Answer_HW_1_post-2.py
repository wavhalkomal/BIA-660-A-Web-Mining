#!/usr/bin/env python
# coding: utf-8

# # <center>HW 1 Analyze Vocabulary and DTM</center>

# <div class="alert alert-block alert-warning">Each assignment needs to be completed independently. Never ever copy others' work (even with minor modification, e.g. changing variable names). Anti-Plagiarism software will be used to check all submissions. </div>

# **Instructions**: 
# - Please read the problem description carefully
# - Make sure to complete all requirements (shown as bullets) . In general, it would be much easier if you complete the requirements in the order as shown in the problem description
# - Submit your codes on Canvas
# - Sample output is ONLY for your reference. 

# ## Q1. Define a function to analyze word counts in an input sentence 
# 
# 
# Define a function named `tokenize(text)` which does the following:
# * accepts a sentence (i.e., `text` parameter) as an input
# * splits the sentence into a list of tokens by **space** (including tab, and new line). 
#     - e.g., `Hello, my friend!!! How's everything?` will be split into tokens `["Hello,", "my","friend!!!","How's","everything?"]`  
# * removes the **leading/trailing punctuations or spaces** of each token, if any
#     - e.g., `friedn!!! -> friend`, while `how's` does not change
#     - hint, you can import module *string*, use `string.punctuation` to get a list of punctuations (say `puncts`), and then use function `strip(puncts)` to remove leading or trailing punctuations in each token
# * only keeps tokens with 2 or more characters, i.e. `len(token)>1` 
# * converts all tokens into lower case
# * find the count of each unique token and save the counts as dictionary, i.e., `{hello: 1, my: 1, ...}`
# * returns the dictionary 
#     

# In[22]:


import string
import pandas as pd
import numpy as np
import re

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[23]:


def tokenize(text):
     
    # enter your code

    import string
 
    # Split the sentence by spaces (including tab and new lines)
    tokens = text.split()
    
    # Initialize a dictionary to store the word counts
    word_counts = {}
    
    # Define the punctuation set to strip from tokens
    puncts = string.punctuation

    for token in tokens:
        # Remove leading and trailing punctuation from the token
        token = token.strip(puncts)
        
        # Convert token to lowercase
        token = token.lower()
        
        # Keep only tokens with more than 1 character
        if len(token) > 1:
            # If the token is already in the dictionary, increment its count
            if token in word_counts:
                word_counts[token] += 1
            else:
                word_counts[token] = 1
    
    return word_counts


# In[24]:


# test your code
text = """I am going to school now. 
          I didn't have lunch today.
          I am flying to LA tomorrow!!!
          I showed my homework to the teacher."""
tokenize(text)


# ## Q2. Generate the vocabulary for a list of sentences.
# 
# - accepts **a list of sentences**, i.e., `sents`, as an input, this will be your corpus
# - uses `tokenize` function you defined in Q1 to get the count dictionary for each sentence
# - build a large vocabulary for all sentences (entire corpus), where the keys are the unique words, and the values are the counts for these words in all the sentences.
# - sort the large dictionary by word count in descending order
# - return the large vocabulary for this corpus

# In[26]:


def generate_vocab(sents):
    # enter your code
 
    # Initialize an empty dictionary to store the vocabulary
    vocab = {}
    
    # Iterate through each sentence in the list
    for sentence in sents:
        # Tokenize the sentence to get the count dictionary
        tokenized = tokenize(sentence)
        
        # Update the vocab with counts from the current sentence
        for word, count in tokenized.items():
            if word in vocab:
                vocab[word] += count
            else:
                vocab[word] = count
    
    # Sort the dictionary by count in descending order
    sorted_vocab = dict(sorted(vocab.items(), key=lambda item: item[1], reverse=True))
    
    return sorted_vocab


    


# In[27]:


# A test document. 

sents = pd.read_csv("sentences.csv", encoding='utf-8')
sents


# In[28]:


# test this function

generate_vocab(sents.sent)


# ## Q3. Generate a document term matrix (DTM) as a numpy array
# 
# 
# Define a function `get_dtm(sents)` as follows:
# - accepts a list of sentences, i.e., `sents`, as an input
# - call `tokenize` function you defined in Q1 to get the count dictionary for each sentence, and combine them into a list
# - call `generate_vocab` function in Q2 to generate the large vocabulary for all sentences, and get all the words, i.e., keys
# - creates a numpy array, say `dtm` with a shape (# of docs x # of unique words), and set the initial values to 0.
# - fills cell `dtm[i,j]` with the count of the `j`th word in the `i`th sentence. HINT: you can loop through the list of vocabulary from step 2, and check each word's index in the large vocabulary from step 3, so that you can put the corresponding value into the correct cell. 
# - returns `dtm` and `unique_words`

# In[30]:


import numpy as np

def get_dtm(sents):
    # Step 1: Tokenize each sentence to get count dictionaries
    count_dicts = [tokenize(sentence) for sentence in sents]  # Assuming sents is a list of sentences

    # Step 2: Generate the vocabulary
    unique_words = generate_vocab(sents)  # Ensure this returns a list of unique words

    # Step 3: Initialize the DTM
    num_docs = len(sents)
    num_words = len(unique_words)
    dtm = np.zeros((num_docs, num_words), dtype=int)

    # Step 4: Fill the DTM
    for i, count_dict in enumerate(count_dicts):
        for word, count in count_dict.items():
            if word in unique_words:
                j = unique_words.index(word)  # Find the index of the word in the unique_words list
                dtm[i, j] = count  # Set the count in the DTM

    return dtm, unique_words


 

print("----executed...------")    
 


# In[31]:


dtm, all_words = get_dtm(sents.sent)


# In[ ]:


# Check if the array is correct
# randomly check one sentence
idx = 3

# get the dictionary using the function in Q1
vocab = tokenize(sents["sent"].loc[idx])
print(sorted(vocab.items(), key = lambda item: item[0]))

# get all non-zero entries in dtm[idx] and create a dictionary
# these two dictionaries should be the same
sents.loc[idx]
vocab1 ={all_words[j]: dtm[idx][j] for j in np.where(dtm[idx]>0)[0]}
print(sorted(vocab1.items(), key = lambda item: item[0]))


# ## Q4 Analyze DTM Array 
# 
# 
# **Don't use any loop in this task**. You should use array operations to take the advantage of high performance computing.

# Define a function named `analyze_dtm(dtm, words)` which:
# * takes an array $dtm$ and $words$ as an input, where $dtm$ is the array you get in Q3 with a shape $(m \times n)$, and $words$ contains an array of words corresponding to the columns of $dtm$.
# * calculates the sentence frequency for each word, say $j$, e.g. how many sentences contain word $j$. Save the result to array $df$ ($df$ has shape of $(n,)$ or $(1, n)$).
# * normalizes the word count per sentence: divides word count, i.e., $dtm_{i,j}$, by the total number of words in sentence $i$. Save the result as an array named $tf$ ($tf$ has shape of $(m,n)$).
# * for each $dtm_{i,j}$, calculates $tf\_idf_{i,j} = \frac{tf_{i, j}}{df_j}$, i.e., divide each normalized word count by the sentence frequency of the word. The reason is, if a word appears in most sentences, it does not have the discriminative power and often is called a `stop` word. The inverse of $df$ can downgrade the weight of such words. $tf\_idf$ has shape of $(m,n)$
# * prints out the following:
#     
#     - the total number of words in the document represented by $dtm$
#     - the most frequent top 10 words in this document, is the result the same as Q2, if not, why? Please explain in text.
#     - words with the top 10 largest $df$ values (show words and their $df$ values)
#     - the longest sentence (i.e., the one with the most words)
#     - top-10 words with the largest $tf\_idf$ values in the longest sentence (show words and values) 
# * returns the $tf\_idf$ array.
# 
# 
# 
# Note, for all the steps, **do not use any loop**. Just use array functions and broadcasting for high performance computation.

# In[ ]:


import numpy as np
import pandas as pd
def analyze_dtm(dtm, words, sents):
    
    # enter your code

    #def analyze_dtm(dtm, words):
    # Step 1: Calculate sentence frequency for each word
    sentence_frequency = np.sum(dtm > 0, axis=0)  # Shape: (num_words,)
    
    # Step 2: Normalize the word count per sentence
    total_words_per_sentence = np.sum(dtm, axis=1)  # Shape: (num_docs,)
    normalized_counts = dtm / total_words_per_sentence[:, np.newaxis]  # Broadcasting to shape (num_docs, num_words)
    
    # Step 3: Calculate the inverse document frequency
    idf = normalized_counts / sentence_frequency  # Broadcasting: (num_docs, num_words)
    
    # Step 4: Print total number of words in the document
    total_words = np.sum(dtm)
    print(f"Total number of words in the document: {total_words}")
    
    # Step 5: Get the most frequent top 10 words
    word_counts = np.sum(dtm, axis=0)  # Total counts for each word
    top_10_frequent_words = np.argsort(word_counts)[-10:][::-1]  # Indices of the top 10 words
    print("Most frequent top 10 words:")
    for idx in top_10_frequent_words:
        print(f"{words[idx]}: {word_counts[idx]}")

    # Step 6: Compare with Q2 result (if applicable)
    # You would need to check against your previous results here.
    
    # Step 7: Get words with the top 10 largest idf values
    top_10_idf_words = np.argsort(idf, axis=None)[-10:][::-1]  # Indices of the top 10 idf values
    top_10_idf_values = idf.flatten()[top_10_idf_words]
    print("Top 10 words with the largest idf values:")
    for idx in top_10_idf_words:
        word_idx = idx % dtm.shape[1]  # Mapping back to word index
        print(f"{words[word_idx]}: {top_10_idf_values[idx]}")
    
    # Step 8: Find the longest sentence
    lengths = np.sum(dtm > 0, axis=1)  # Count non-zero entries in each row (sentence length)
    longest_sentence_idx = np.argmax(lengths)  # Index of the longest sentence
    print(f"Longest sentence index: {longest_sentence_idx}, Length: {lengths[longest_sentence_idx]}")
    
    # Step 9: Top 10 words in the longest sentence based on idf
    longest_sentence_dtm = dtm[longest_sentence_idx]  # Get the DTM for the longest sentence
    longest_sentence_idf = idf[longest_sentence_idx]  # Get the idf for the longest sentence
    top_10_longest_sentence_words = np.argsort(longest_sentence_idf)[-10:][::-1]
    print("Top 10 words with the largest idf values in the longest sentence:")
    for idx in top_10_longest_sentence_words:
        if longest_sentence_dtm[idx] > 0:  # Check if the word exists in the longest sentence
            print(f"{words[idx]}: {longest_sentence_idf[idx]}")

    return idf

    


# In[ ]:


sents_list = sents["sent"].tolist()  # If sents is a DataFrame
sents_list 
dtm, all_words = get_dtm(sents_list)

words = np.array(all_words)

analyze_dtm(dtm, words, sents.sent)


# ## Bonus. Find keywords of the document (1 point)
# Can you leverage  𝑑𝑡𝑚
#   array you generated to find a few keywords that can be used to tag this document? e.g., AI, ChatGPT, etc.
# 
# Describe your ideas and also implement your ideas.

# In[ ]:





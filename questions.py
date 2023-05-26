import nltk
import sys
import os
import string
import math
from itertools import chain

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = {}
    for filename in os.listdir(directory):
        folder_path = os.path.join(directory,filename)
        with open(folder_path, mode="r", encoding="utf8") as f:
            content = f.read()
            files[filename] = content
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = [] 
    punctuations = string.punctuation
    stopwords = nltk.corpus.stopwords.words("english")
    words.extend([
        word for word in nltk.word_tokenize(document.lower()) if (word not in punctuations and word not in stopwords)
    ])
    return words



def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = {}
    for filename, list_word1 in documents.items():
        for word in list_word1:
                if word not in idfs:
                    count = sum(1 for list_word2 in documents.values() if word in list_word2)
                    idf = math.log(len(documents) / count)
                    idfs[word] = idf
                    
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    files_ranked = {}
    
    for filename , list_word in files.items():
        tf_idfs = {}
        for word in query:
            if word not in tf_idfs and word in idfs and word in list_word:
                idf = idfs[word] 
                tf = sum(1 for item in list_word if word in item)
                tfidf = tf * idf
                tf_idfs[word] = tfidf
            
        files_ranked[filename] = sum(tf_idfs[word] for word in query if word in tf_idfs)         
    
    top_sorted = [k for k, v in sorted(files_ranked.items(), key=lambda x: x[1], reverse=True)]
    return top_sorted[:n]


    
def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentences_ranked = {}
    for key in sentences:
        sum_idfs = 0
        matching_words = 0
        sentence = sentences[key]
        for word in query:
            if word in sentence:
                matching_words += 1
                if word in idfs:
                    sum_idfs += idfs[word]
        
        qtd = matching_words / len(sentence)
        sentences_ranked[key] = (sum_idfs , qtd)
                
    top_score = [k for k, v in sorted(sentences_ranked.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)][:n]
    return top_score



if __name__ == "__main__":
    main()

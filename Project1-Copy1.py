from sklearn.model_selection import train_test_split
import nltk
nltk.download('treebank')
nltk.download('conll2002')
from nltk.corpus import treebank
from nltk.corpus.reader import ConllCorpusReader
from utils import utils

class Project1:

    
    # Programming Question #1
    # Paste your answer for section 2.2.1 Programming Question #1 of Project 1 inside the 
    # function below.
    def word_given_tag(word, tag, train_bag):
        # if train_bag is None:
        #     train_bag = self.train_tagged_words
        """
        This function accepts word, tag and tagged words in training data to return count(w|tag) and count(tag)

        :param word: string
        :param tag: string
        :param train_bag: list of <_word_, _tag_>
        :return: count(w|tag) <integer>, count(tag) <integer>
        """
        # Write approximately 1 to 5 lines of code:

        # find the list (L) of pairs (_word_, _tag_) in train_bag having _tag_ equal to tag.
        L = [(_word_, _tag_) for (_word_, _tag_) in train_bag if _tag_ == tag]
        # find the count of elements in list (L) - total number of times the passed tag occurred in train_bag
        count = len(L)
        # find the number of times the word appears in pair <word, tag> of list (L)
        word_count = len([(_word_, _tag_) for (_word_, _tag_) in L if _word_ == word])



        # return the word count given tag and the count of elements in the list of pairs
        return (word_count, count)


    # Programming question #2
    # Paste your answer for section 2.2.2 Programming Question #2 of Project 1 inside the 
    # function below.
    def t2_given_t1(t2, t1, train_bag):
        # if train_bag is None:
        #     train_bag = self.train_tagged_words
        """
        This function accepts two adjacent tags appearing in the text and tagged words in training data to return the count(t2|t1) and count(t1)

        :param t1: string
        :param t2: string
        :param train_bag: list of <_word_, _tag_>
        :return: count(t2|t1), count(t1)
        """

        # Write approximately 1 to 5 lines of code:

        # find the list (T) of tags present in the pairs of train_bag <_word_, _tag_>
        T = set([_tag_ for (_word_, _tag_) in train_bag])

        # count number of times t1 is present in the List (T)
        T1_count = sum(1 for (_word_, _tag_) in train_bag if _tag_ == t1)

        # count the number of times t2 appears after t1 in the List (T)
        T2_given_T1_count = sum(1 for i in range(len(train_bag)-1) 
                                if train_bag[i][1] == t1 and train_bag[i+1][1] == t2)



        # return count for t2 after t1, and the count for t1
        return (T2_given_T1_count, T1_count)

    # Programming question 3
    # Paste your answer for section 2.3 Programming Question #3 of Project 1 inside the 
    # function below.
    def compute_state_probability(key, word, tags_df, T, state):
        """
        This function accepts key, word, list of tags T, the previous state (tag) and returns the list of probabilities of each tag in T
        being the next state


        :param key: int the position of the word in the sentence
        :param word: string
        :param T: List of unique tags
        :param tag: string
        :param p: List of probabilities for current iteration

        :return: List<state_probabilities>
        """
        word_given_tag = utils.word_given_tag
        p = [] 
        for tag in T:
            if key == 0:
                transition_p = tags_df.loc['.', tag]
            else:
                transition_p = tags_df.loc[state[-1], tag]
            
            # Write approximately 1 to 5 lines of code to compute emission and state probabilities:

            # calculate Emission probabilities
            word_count, tag_count = word_given_tag(word, tag)
            emission_p = word_count / tag_count if tag_count > 0 else 0

            # calculate state probabilities
            state_probability = transition_p * emission_p

            # add state probability in the list
            p.append(state_probability)
            
        return p

    
    # Programming question 4
    # Paste your answer for section 3.1 Programming Question #4 of Project 1 inside the 
    # function below.
    def compute_tag_distribution(sents):
        """
        This function accepts a tuple <word, pos, ner> to return the NER_tag frequency distribution

        :param List of tuple <word, pos, ner>: List of (string, string, string)
        :return ner_tag_frequency: a dictionary of frequency of NER tags.
        """
        ner_tag_frequency=dict()

        # Write approximate 2-10 lines of code to create dictionay of <word, NER_tag> pairs.
        # In a list of sents given as a tuple <word, pos, ner>, count how many words
        # are there for each ner tag:

        for sent in sents:
            for (word, pos, ner) in sent:
                ner_tag_frequency[ner] = ner_tag_frequency.get(ner, 0) + 1

        return ner_tag_frequency


    # Programming question 5
    # Paste your answer for section 3.2 Programming Question #5 of Project 1 inside the 
    # function below.
    def word2features(sent, i):
        """
        This function accepts a sent (list of tuple in sentence) to return the additional features.

        :param sent: List of (string, string, string)
        :return features: dictionary
        """

        word = sent[i][0]
        postag = sent[i][1]
        
        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[:2],        
        }
        if i > 0:
            

            #find (i) previous word and (ii) postag for previous word.
            word1 = sent[i-1][0]
            postag1 = sent[i-1][1]

            # Add the following features for the previous word:
            # lowercased version of the word
            # value indicating whether word is title
            # value indicating whether word is uppercase
            # POS tag of the word
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True
            
        if i < len(sent)-1:

            # Write approximately 8-10 lines of code to add features for the next word.
            #find (i) next word and (ii) postag for next word.
            word2 = sent[i+1][0]
            postag2 = sent[i+1][1]

            # Add the following features for the next word:
            # lowercased version of the word
            # value indicating whether word is title
            # value indicating whether word is uppercase
            # POS tag of the word
            features.update({
                '+1:word.lower()': word2.lower(),
                '+1:word.istitle()': word2.istitle(),
                '+1:word.isupper()': word2.isupper(),
                '+1:postag': postag2,
                '+1:postag[:2]': postag2[:2],
            })
        else:
            features['EOS'] = True
                    
        return features
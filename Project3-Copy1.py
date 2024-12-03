import pandas as pd
import numpy as np
from pandas import DataFrame
from transformers import BertTokenizer

import torch


class Project3:

    # Programming question 1
    # @title Complete code here to predict semantic roles for active and passive sentences
    # You will define srl_prediction below to return a data frame using Pandas DataFrame
    def srl_prediction(model,sent:str) -> DataFrame :

        """
        :param model : en_srl_predictor
        :param sent : string
        :return df : dataframe
        """
    
        '''
        Coding Instrution :

            1. Use model.predict to predict json format SRL prediction
            2. Return data frame using Pandas DataFrame
        '''

        #Write your code here (4-5 lines)
        result = model.predict(sent)
        verbs = result.get('verbs', [])
        data = []
        for verb in verbs:
            data.append({
                'verb': verb['verb'],
                'description': verb['description']
            })
        df = pd.DataFrame(data)
        #Your code ends here
        print(df['description'])

        return df

    # Programming question 2
    #@title Complete code here to generate BertTokenizer
    def get_bert_tokens(sent:str) -> list :

        """
        :param sent : string
        """
        '''
        Coding Instruction :
            1 Build BertTokenizer using from_pretrained model : "bert-base-uncased"
            2.tokenize sentence(input) using tokenizer
            3.convert new input ids to token (use convert_ids_to_tokens)
            4.return list of tokens
        '''

        #Write your code here (3-4 lines)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        encoded_sent = tokenizer.encode(sent)
        token_list = tokenizer.convert_ids_to_tokens(encoded_sent)
        #End your code here
        print(token_list)
        return token_list

    # Programming question 3
    #@title Complete this cell to predict Mask in the sentence.
    def masked_learning(tokenizer,model, country) :
        """
        :param country : string
        :param tokenizer : BertTokenizer
        :param model : BertForMaskedLM

        """

        '''
        Programming Instruction :
        You need to complete three parts : #complete_here[1], #complete_here[2], and #complete_here[3]
        '''

        question =  "The capital of "+country+" is [MASK]."

        inputs = tokenizer(question, return_tensors = "pt" ) 

        with torch.no_grad():
            # complete code that return logit(non-normalized probabilites) of model's output
            logits = model(**inputs).logits
        
        index_for_masked_token = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple = True)[0]
        # complete code that return maximum values of a tensor
        id_for_predicted_token = logits[0,index_for_masked_token].argmax(dim=-1)

        #complete function name to decode tensor(id_for_prediction_token) to string
        city = tokenizer.decode(id_for_predicted_token)
        
        print(city)
        return "The capital of {} is {}".format(country, city)

    # Programming question 4
    #@title Complete this code below
    def get_srl_objects(predictor, sentences) :
        objs_list = []

        '''
        Coding Instruction :
        1. Predict SRL using fr_srl_predictor, en_srl_predictor
        2. lang_object dictionary, and add key-'sent', value - sentence
        3. After adding sentence, append each lang_obj into objs_list
        '''
        #your code start from here (3~5 lines)
        for sentence in sentences:
            lang_obj = predictor.predict(sentence)
            lang_obj['sent'] = sentence
            objs_list.append(lang_obj)
        #end your code here
        print(objs_list)

        return objs_list
    
    # Programming question 5
    #@title Complete code here to return french_sentence, english_sentence
    def get_tokens(fr_obj, en_obj) -> tuple:

        '''
        Programming instruction :
          1. Get tokens of French sentence and Englsih sentence
          2. sentence1 : French sentence tokens
          3. sentence2 : English sentence tokens
        '''
        #Write your code here(2-3 lines)
        french_tokens = fr_obj['words']
        english_tokens = en_obj['words']
        #End code here

        return (french_tokens, english_tokens)
    
    # Programming question 6
    #@title Complete this code to return Precision, Recall, and F-1 Score
    def precision_recall_f1(confusion_matrix:dict) -> tuple :

        '''
        Programming Instruction
          1. Write code to return precision recall and f1 using defined formula above
          2. Consider this case : If denominator is 0, precision, recall and f1 score should be 0
        '''

        #Write your code start here(15-20 line)
        # precision
        if confusion_matrix['TP'] + confusion_matrix['FP'] > 0:
            precision = confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FP'])
        else:
            precision = 0

        # recall
        if confusion_matrix['TP'] + confusion_matrix['FN'] > 0:
            recall = confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FN'])
        else:
            recall = 0

        # F1 score
        if precision + recall > 0:
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0
        #Your code end here
        
        return (precision, recall, f1)


import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pickle
import os


corpus_list_data = os.path.join('embedding-data/', 'corpus_list_data.pickle')
corpus_embeddings_data = os.path.join('embedding-data/', 'corpus_embeddings_data.pickle')
corpus_with_diacritics = os.path.join('embedding-data/', 'corpus_with_diacritics.pickle')


class TextEmbedder:
    def __init__(self):
        self.corpus_embeddings_data = corpus_embeddings_data
        self.corpus_list_data = corpus_list_data
        self.corpus_list = None
        self.embedder = SentenceTransformer('UBC-NLP/ARBERT')
        self.corpus_embeddings = None
        self.corpus_with_diacritics = corpus_with_diacritics
        if 'embedding-data' not in os.listdir():
            os.makedirs("embedding-data")
    def embed(self,corpus_list:list, df_diacritics):
        self.corpus_list = corpus_list
        if len(os.listdir("embedding-data/"))==0:
            self.corpus_embeddings = self.embedder.encode(self.corpus_list,show_progress_bar=True)
            df1 = pd.DataFrame()
            df1['features'] = list(self.corpus_embeddings)
            df2 = pd.DataFrame()
            df2['sentences'] = self.corpus_list
            pickle.dump(df1, open(self.corpus_embeddings_data, "wb"))
            pickle.dump(df2, open(self.corpus_list_data, "wb"))
            pickle.dump(df_diacritics, open(self.corpus_with_diacritics, "wb"))
            
            print("Embedding data Saved Successfully!")
            print(os.listdir("embedding-data/"))
        else:
            print("Embedding data allready present, Do you want Embed & Save Again? Enter yes or no")
            flag  = str(input())
            if flag.lower() == 'yes':
                self.corpus_embeddings = self.embedder.encode(self.corpus_list,show_progress_bar=True)
                df1 = pd.DataFrame()
                df1['features'] = list(self.corpus_embeddings)
                df2 = pd.DataFrame()
                df2['sentences'] = self.corpus_list
                pickle.dump(df1, open(self.corpus_embeddings_data, "wb"))
                pickle.dump(df2, open(self.corpus_list_data, "wb"))
                pickle.dump(df_diacritics, open(self.corpus_with_diacritics, "wb"))
                print("Embedding data Saved Successfully Again!")
                print(os.listdir("embedding-data/"))
            else:
                print("Embedding data allready Present, Please Apply Search!")
                print(os.listdir("embedding-data/"))
    def load_embedding(self):
        if len(os.listdir("embedding-data/"))==0:
            print("Embedding data Not present, Please Run Embedding First")
        else:
            print("Embedding data Loaded Successfully!")
            print(os.listdir("embedding-data/"))
            return pickle.load(open(self.corpus_embeddings_data, "rb"))

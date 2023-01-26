import os
import pandas as pd
import numpy as np
from annoy import AnnoyIndex
import logging
logger = logging.getLogger(__name__)
from DeepTextSearch import TextEmbedder

class SearchImage:
    def __init__(self, corpus_path, index_path):
        self.corpus_path = corpus_path
        self.index_path = index_path
        self.image_data = pd.read_pickle(self.corpus_path)
        self.feature_vector_length = 768 # Give us the length of the feature vector

    def search_by_vector(self, vector, n_out: int):
        """This function will search for the most similar images for the given vector and time it takes is (0.000171661376953125 Seconds)"""
        index_s = AnnoyIndex(self.feature_vector_length, 'euclidean')
        index_s.load(self.index_path)  # superfast, will just mmap the file
        index_list = list(index_s.get_nns_by_vector(vector, n_out, include_distances=True)) # will find the 10 nearest neighbors
        # Map the index_list[1] from float to string
        #index_list[1] = list(map(str, index_list[1]))
        # the index list output is like that ([6, 7, 8], [0.0, 0.6026723384857178, 0.8105881214141846, 0.8481426239013672])
        # index_list[1] it means the scores
        # index_list[0] it gives us the product ids
        sim = dict(zip(self.image_data.iloc[index_list[0]]['sentences'].to_list(), index_list[1]))
        sorted_tuples = sorted(sim.items(), key=lambda item: item[1])
        ids = [i[0] for i in sorted_tuples]
        scores = [i[1] for i in sorted_tuples]
        #similar_images = {'sim_product_ids': ids, 'scores': scores}
        similar_images = {k: round(v, 4) for k, v in sorted_tuples}
        return similar_images, ids, scores


    def get_similar_quotes(self, query_text: str, number_of_images: int):
        query_vector = TextEmbedder().embedder.encode(query_text, device="cpu") # the feature vector of the input image
        if str(type(query_vector)) != "<class 'numpy.ndarray'>":
            """ Sometimes we have duplicates in images and the same loc return 2 vectors so we handle them by this if condition"""
            sim_tuple = self.search_by_vector(query_vector[0], number_of_images)
        else:
            sim_tuple = self.search_by_vector(query_vector, number_of_images)
        sim_prod_item = {'qoute': query_text,
                         'similar_qoutes': sim_tuple[0], "scores": sim_tuple[2] }
        return sim_prod_item


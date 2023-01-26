import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from annoy import AnnoyIndex
from tqdm import tqdm

class GetIndexFiles:
    def __init__(self, embedding_file_path = None, asr_name = None):
        self.embedding_file_path = embedding_file_path
        self.asr_name = asr_name

    def start_indexing(self):
        """This function will start the indexing process for the given department"""

        text_features = pd.read_pickle(self.embedding_file_path)
        feature_vector_length = len(text_features['features'][0])  # Length of item vector that will be indexed
        index = AnnoyIndex(feature_vector_length, 'euclidean')
        for i, v in tqdm(zip(text_features.index, text_features['features'])):
            index.add_item(i, v)
        index.build(100)  # 100 trees
        print(f"Saved the Indexed File:[meta-data/text_features_index.ann]")
        index.save(f"meta_data/text_features_index_{self.asr_name}.ann")




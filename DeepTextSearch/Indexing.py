import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from annoy import AnnoyIndex
from tqdm import tqdm
class GetIndexFiles:
    def __init__(self):
        self.image_data = None

    def start_indexing(self):
        """This function will start the indexing process for the given department"""

        text_features = pd.read_pickle("embedding-data/corpus_embeddings_data.pickle")
        feature_vector_length = len(text_features['features'][0])  # Length of item vector that will be indexed
        index = AnnoyIndex(feature_vector_length, 'euclidean')
        for i, v in tqdm(zip(text_features.index, text_features['features'])):
            index.add_item(i, v)
        index.build(100)  # 100 trees
        print(f"Saved the Indexed File:[meta-data/text_features_index.ann]")
        index.save(f"meta_data/text_features_index.ann")

index = GetIndexFiles()
index.start_indexing()


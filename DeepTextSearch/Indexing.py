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

        image_data = pd.read_pickle("/media/youssef/DVolume/AI/home/upwork/eqtibas/embedding-data/corpus_embeddings_data.pickle")
        feature_vector_length = len(image_data['features'][0])  # Length of item vector that will be indexed
        index = AnnoyIndex(feature_vector_length, 'euclidean')
        for i, v in tqdm(zip(image_data.index, image_data['features'])):
            index.add_item(i, v)
        index.build(100)  # 100 trees
        print(f"Saved the Indexed File:[meta-data/text_features_index.ann]")
        index.save(f"/media/youssef/DVolume/AI/home/upwork/eqtibas/meta_data/text_features_index.ann")

index = GetIndexFiles()
index.start_indexing()

from DeepTextSearch import TextEmbedder
import pandas as pd
import re

poem_data = pd.read_csv("data/Arabic Poem Comprehensive Dataset (APCD).csv")
df_hadith = poem_data[(poem_data['العصر'] == 'الحديث')]

verses = list(df_hadith['البيت'])
# Load data from CSV file
def remove_diacritics(text: str) -> str:
    # Remove tashkeel
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)

    # Remove tatweel
    text = re.sub(r'\u0640', '', text)

    return text


cleaned_poet = []
for x in verses:
    cleaned_verse = remove_diacritics(x)
    cleaned_poet.append(cleaned_verse)

# To use Searching, we must first embed data. After that, we must save all of the data on the local path.
t_embedder = TextEmbedder()

t_embedder.embed(corpus_list=cleaned_poet)




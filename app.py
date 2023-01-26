import streamlit as st
#from DeepTextSearch import Searching
import Searching
text = " فضل المدرس عاى الطالب كبير جدا لا يجب نكرانه"
#result = search.find_similar(query_text=tweet ,top_n=5)
#for sentence in result:
#    print(sentence)

arab_eras = {'Before_Islam' : 'قبل الإسلام', 'Veteran':'المخضرمين', 'Abbasid':'العباسي',
 'Umayyad': 'الأموي', 'Mamluk':'المملوكي', 'Morocco_and_Andalusia':'المغرب والأندلس',
 'Between_the_Two States':'بين الدولتين', 'Fatimid':'الفاطمي', 'Ayyubid':'الأيوبي',
 'Hadith':'الحديث', 'Islamic':'الإسلامي', 'Ottoman':'العثماني'}
arab_eras_inv = {v: k for k, v in arab_eras.items()}


st.header('Eqtibas :sunglasses:')

col1, col2 = st.columns(2)

with col1:
    with_diacritics = st.selectbox(
        "هل تريد الشعر بالتشكيل؟ ",
        ("نعم", "لا"),
    )

with col2:
    era = st.selectbox(
        "اختر العصر الشعرى الذي تريده",
        tuple(arab_eras.values()),
    )

text = st.text_area(" :) اكتب ما يدور في بالك")

if st.button("Process Text"):
    # process text and display output
    if with_diacritics == 'نعم':
        courps_path = f"embedding-data/corpus_with_diacritics_{arab_eras_inv[era]}.pickle"
    else:
        courps_path = f"embedding-data/corpus_list_data_{arab_eras_inv[era]}.pickle"
    ind_path = f"meta_data/text_features_index_{arab_eras_inv[era]}.ann"
    search = Searching.SearchImage(courps_path, ind_path)

    results = search.get_similar_quotes(text, 5)
    for result in results['similar_qoutes'].keys():
        result = result.replace('    ', '  ***  ')
        print(result)
        st.markdown(f"> {result}")

    del search

# replace the spaces from two sentences like this مكارم الأخلاق فيهم لا تحد    فحبهم فرض على كل أحد  to *


import streamlit as st
from DeepTextSearch import Searching

text = " فضل المدرس عاى الطالب كبير جدا لا يجب نكرانه"
#result = search.find_similar(query_text=tweet ,top_n=5)
#for sentence in result:
#    print(sentence)
st.header('Eqtibas :sunglasses:')

search = Searching.SearchImage()
text = st.text_area("Enter a block of text:")


if st.button("Process Text"):
    # process text and display output
    results = search.get_similar_quotes(text, 5)
    for result in results['similar_qoutes'].keys():
        result = result.replace('    ', '  ***  ')
        print(result)
        st.markdown(f"> {result}")

# replace the spaces from two sentences like this مكارم الأخلاق فيهم لا تحد    فحبهم فرض على كل أحد  to *


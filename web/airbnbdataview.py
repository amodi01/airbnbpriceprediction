import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import pickle5 as pickle

st.title("Airbnb NYC Data Jan-2020- Dec2-020")
#File Location Variables
_LocationName="NY"
_DataFolderPath="raw_data_csv"
_LocationPath=_DataFolderPath +"/" + _LocationName
_PickleFilesFolder="pickle_files"
_PickleFile_Merged_Listing_NY="Merged_Listing_NY_4"

#Reading Pickle File
@st.cache
def load_data(nrows):
    with open(_PickleFile_Merged_Listing_NY,'rb') as f:
        _DF_LISTING_EDA = pickle.load(f)
    return _DF_LISTING_EDA.head(nrows)

data_load_state=st.text("Loading Data....")
data=load_data(100)
data_load_state.text("Loading Data ..Done!!")

is_check = st.checkbox("Display Data")
if is_check:
    st.write(data)


st.bar_chart(data.groupby('property_type')['price'].mean())

st.bar_chart(data.groupby('room_type')['price'].mean())


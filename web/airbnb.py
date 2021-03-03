import numpy as np
import pandas as pd
import pickle5 as pickle
import pandas as pd
import streamlit as st
import plotly.express as px
import SessionState
import preditc
my_page = st.sidebar.radio('Select Option Below :', ['Data Visualization', 'Predictions','Raw Data'])

st.title("NewYork Airbnb (EDA and Predictions)")
st.markdown("Data Collected from Air Bnb website for the period of Jan-2020 To Dec-2020")

data_load_state = st.text("")
#st.balloons()

# Region Starts Global Methods
# Reading Pickle File

def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

# Region Ends Global Methods

# Region Starts Global Variables
_DataFolderPath = "data"
_PickleFile_Merged_Listing_NY_4 = "Merged_Listing_NY_4"
_PickleFile_Merged_Listing_NY = "Merged_Listing_NY"
_LocationPath = _DataFolderPath + "/" + _PickleFile_Merged_Listing_NY
_LocationPath_4=_DataFolderPath + "/" + _PickleFile_Merged_Listing_NY_4
_Columns_List=['name','description','neighbourhood','neighborhood_overview','host_name','host_since','host_response_rate']
# Number of entries per screen
_Rows_per_Page = 50
# A variable to keep track of which product we are currently displaying
_Session_State = SessionState.get(page_number = 0)
data_load_state.text("Loading Data ..Done!!")
_DF_Comeplete_Data_4 = load_data(_LocationPath_4)
_DF_Comeplete_Data = load_data(_LocationPath)
data_load_state.text("Loading Data ..Done!!")
data_load_state.text("")
_Last_page = len(_DF_Comeplete_Data) // _Rows_per_Page

# Region Ends Global Variables
if my_page == 'Raw Data':
    st.title("Raw Data")
    prev, _, next = st.beta_columns([1, 10, 1])
    if next.button("Next"):

        if _Session_State.page_number + 1 > _Last_page:
            _Session_State.page_number = 0
        else:
            _Session_State.page_number += 1

    if prev.button("Previous"):

        if _Session_State.page_number - 1 < 0:
            _Session_State.page_number = _Last_page
        else:
            _Session_State.page_number -= 1
    start_idx = _Session_State.page_number * _Rows_per_Page
    end_idx = (1 + _Session_State.page_number) * _Rows_per_Page
    sub_df = _DF_Comeplete_Data.iloc[start_idx:end_idx]

    st.write(sub_df[_Columns_List])

elif my_page == 'Data Visualization':
    st.sidebar.title('EDA And Predictions')
    st.sidebar.markdown('Interact with the data here')
    values = st.sidebar.slider("Price range", float(_DF_Comeplete_Data_4.price.min()), 1000., (1., 300.))
    f = px.histogram(_DF_Comeplete_Data_4.query(f"price.between{values}"), x="price", nbins=15, title="Price Distribution")
    f.update_xaxes(title="Price")
    f.update_yaxes(title="No. of listings")
    st.plotly_chart(f)

    is_room_type= st.sidebar.checkbox("Average Price By Room type")
    if is_room_type:
        st.markdown("Average Price By Room Type")
        #st.table(_DF_Comeplete_Data.groupby("room_type").price.mean().reset_index() \
         #        .round(2).sort_values("price", ascending=False) \
          #       .assign(avg_price=lambda x: x.pop("price").apply(lambda y: "%.2f" % y)))

        st.bar_chart(_DF_Comeplete_Data_4.groupby('room_type')['price'].mean())

    is_nhood_type= st.sidebar.checkbox("Average Price by Property Type")

    if is_nhood_type:
        st.markdown("Average Price By Property Type")
        st.bar_chart(_DF_Comeplete_Data_4.groupby('property_type')['price'].mean())
else:
    st.sidebar.markdown('Make Predictions')
    opt = st.sidebar.radio("Select One", ["Load Data", "Manual Input Data"])

    if opt=="Load Data":
        values = st.sidebar.slider("Data range (Record Count)", 1., float( len(_DF_Comeplete_Data)), (1., 1.))
        print(values)
        min=int(values[0])
        max=int(values[1])
        #data_load_state_f = st.text("Loading Data... Please wait...")
        df = _DF_Comeplete_Data.iloc[min:max]
        st.write(df[_Columns_List])
        st.markdown("Data Loaded from range : " + str(min) +  " : " + str(max) + "Records")
        if st.button("Predict"):
            #result = prediction(Gender, Married, ApplicantIncome, LoanAmount, Credit_History)
            dataForPrediction = _DF_Comeplete_Data.iloc[min:max]
            result= preditc.predict(dataForPrediction)
            #data_frames = [dataForPrediction,pd.DataFrame( result)]
            st.sidebar.markdown('Below Are The  Predictions For Selected Recrds : ')
            st.write(result)
            st.balloons()
            #st.success('Your loan is {}'.format(result))
            #print(LoanAmount)
    else:

         sentence = st.text_input('Input your sentence here:')
#https://towardsdatascience.com/streamlit-101-an-in-depth-introduction-fc8aad9492f2
#https://medium.com/@ansjin/how-to-create-and-deploy-data-exploration-web-app-easily-using-python-a03c4b8a1f3e

#apr_avg_adjusted_price,host_age,accommodates,amenties_count,description_count,availability_365,neighborhood_overview_len,bathrooms_text,host_acceptance_rate,host_listings_count
#apr_avg_adjusted_price,host_age,accommodates,amenties_count,description_count,availability_365,neighborhood_overview_len,bathrooms_text,host_acceptance_rate,host_listings_count
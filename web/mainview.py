import streamlit as st
import numpy as np
import pandas as pd
import pickle5 as pickle
import pandas as pd
import streamlit as st
import plotly.express as px
import SessionState

from multiapp import MultiApp

# from apps import home, data_stats # import your app modules here
st.title("NewYork Airbnb (EDA and Predictions)")
st.markdown("Data Collected from Air Bnb website for the period of Jan-2020 To Dec-2020")
st.sidebar.title('EDA And Predictions')
data_load_state = st.text("")
st.balloons()
import numpy as np
import pandas as pd
import pickle5 as pickle
import pandas as pd
import streamlit as st
import plotly.express as px
import SessionState

st.title("NewYork Airbnb (EDA and Predictions)")
st.markdown("Data Collected from Air Bnb website for the period of Jan-2020 To Dec-2020")
st.sidebar.title('EDA And Predictions')
data_load_state = st.text("")
st.balloons()


# Region Starts Global Methods
# Reading Pickle File

def load_data():
    with open(_LocationPath, 'rb') as f:
        data = pickle.load(f)
    return data


# Region Ends Global Methods

# Region Starts Global Variables
_DataFolderPath = "data"
_PickleFile_Merged_Listing_NY = "Merged_Listing_NY_4"
_LocationPath = _DataFolderPath + "/" + _PickleFile_Merged_Listing_NY

app = MultiApp()

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Data Stats", data_stats.app)

# The main app
app.run()

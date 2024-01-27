import streamlit as st
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import ordpy
from PIL import Image

# Title


# image = Image.open("D:/Tesis/dog.png")

# st.image(image, width=500)

st.title("Theory Information Cuantifiers")
st.markdown(
    """
This app retrieves currency prices from the **yahoo finance web API.**
    """
)


# ---------------------------------#
# About

st.markdown(
    """
* **Python libraries:** streamlit, pandas, requests, numpy, matplotlib, seaborn, scipy.stats, ordpy, pillow
* **Data source:** [Yahoo Finance API](https://cryptocointracker.com/yahoo-finance/yahoo-finance-api).
* **References** [Shannon, 1948](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf), [Band and Pompe, 2002](https://ri.conicet.gov.ar/handle/11336/99331).
* **Carlos Ibáñez Acuña. -> GitHub https://github.com/cibaneza
    """
)
# ---------------------------------#

st.markdown(
    """
    ## Select your favorite stock! 
    """
)

# Initial UI
ticker = st.text_input("Ticker", "NFLX").upper()
buttonClicked = st.button("Start")

# Callbacks

st.markdown(
    """
    ## Data main information
    """
)

if buttonClicked:
    try:
        requestString = f"""https://query1.finance.yahoo.com/v11/finance/quoteSummary/{ticker}?modules=assetProfile%2Cprice"""
        request = requests.get(
            f"{requestString}", headers={"USER-AGENT": "Mozilla/5.0"}
        )
        json = request.json()
        data = json["quoteSummary"]["result"][0]

        st.header("Profile")

        st.metric("sector", data["assetProfile"]["sector"])
        st.metric("industry", data["assetProfile"]["industry"])
        st.metric("website", data["assetProfile"]["website"])
        st.metric("marketCap", data["price"]["marketCap"]["fmt"])

        with st.expander("About Company"):
            st.write(data["assetProfile"]["longBusinessSummary"])
    except:
        pass

data = yf.download(ticker)
# data.head(10)
# Export Data Buttom
st.dataframe(data)


@st.cache_data(ttl=600)
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")


csv = convert_df(data)
st.download_button(
    label="Download data as .csv",
    data=csv,
    file_name=f"{ticker}.csv",
    mime="text/csv",
)


# Plots
st.markdown(
    """
    ### Adjusted Close Price
    """
)
st.line_chart(data["Adj Close"])
st.markdown(
    """
    ### Returns
    """
)
st.line_chart(data["Adj Close"].pct_change())
st.markdown(
    """
    ### Volume
    """
)
st.line_chart(data["Volume"])


st.markdown(
    """
    ## Fisher & Entropy
    """
)

# Fisher & Shannon window

data_daily_returns = data["Adj Close"].pct_change()

# We define the size of the window (250 days in our case, trying to emulate months continously)
# window_size = 250
# window_size = st.text_input("Window Size", "252").upper()

st.markdown(
    """
    ### Select your Window Size!
    """
)

window_size = st.number_input("Insert a Window Size", 252)

st.text("Use 252 as 252 days window size")

# warehouse, here we are saving our results of each window
fisher_info = []

for i in range(len(data_daily_returns) - window_size + 1):
    # Get the window of returns
    window = data_daily_returns[i : i + window_size]

    # Calculate the Fisher information
    fisher_shannon_entropy = ordpy.fisher_shannon(window, dx=2)

    # Get the corresponding date for the window
    window_dates = data.index[i : i + window_size]
    window_date = window_dates[-1]  # Use the last date in the window

    # Create a dictionary with the date and Fisher information
    window_info = {"Date": window_date, "Fisher Information": fisher_shannon_entropy}

    # Add the dictionary to the list
    fisher_info.append(window_info)

# Convert the list to a DataFrame
fisher_info_df = pd.DataFrame(fisher_info)

# print(fisher_info_df)
# st.dataframe(fisher_info_df)

df = pd.DataFrame(fisher_info_df)
df2 = df
df2[["Shannon Entropy", "Fisher Information"]] = df2["Fisher Information"].apply(
    pd.Series
)
df3 = df2
df3.set_index("Date", inplace=True)
# df3.head()

st.dataframe(df3)

st.markdown(
    """
    ### Fisher Information
    """
)
st.line_chart(df3["Fisher Information"])

st.markdown(
    """
    ### Shannon Entropy
    """
)
st.line_chart(df3["Shannon Entropy"])

# fig, ax = plt.subplots()
# plt.plot(df3["Shannon Entropy"])
# st.pyplot(fig)

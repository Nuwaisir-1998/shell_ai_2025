import streamlit as st
import pandas as pd
import numpy as np

conn = st.connection("my_database")
df = conn.query("select * from my_table")
st.dataframe(df)
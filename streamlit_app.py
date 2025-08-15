import streamlit as st

# Define the pages
main_page = st.Page("main.py", title="Main Page")
train_data_page = st.Page("train_data.py", title="Train Data")
test_data_page = st.Page("test_data.py", title="Test Data")

# Set up navigation
pg = st.navigation([main_page, train_data_page, test_data_page])

# Run the selected page
pg.run()
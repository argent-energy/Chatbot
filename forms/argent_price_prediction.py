import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import math
from langchain.prompts import PromptTemplate
from utils.langchain_utils import create_llm_chain_argent

df=pd.read_csv("Gas and Oil Prices.csv")

selected_columns=df[['Date','Argent Gas/Oil Category','Argent Gas/Oil Price Name','Price EUR/mt']]
week,month,quarter=[],[],[]
df = df[df['Source Gas/Oil Price Name'] == 'Biodiesel FAME 0C CFPP RED ARA range barge fob_midpoint']
for index,row in df.iterrows():

        date = pd.to_datetime(row[0], format='%d/%m/%Y')

        first_day_of_month = date.replace(day=1)


        week_of_month = math.ceil((date.day + first_day_of_month.weekday()) / 7)
        week.append(week_of_month)
        month.append(date.month)
        quarter.append(date.quarter)
df['Week of month']=week
df['Month']=month
df['Quarter']=quarter

df['Price_7']= df['Price EUR/mt'].shift(7)
df['Price_6']= df['Price EUR/mt'].shift(6)
df['Price_5']= df['Price EUR/mt'].shift(5)
df['Price_4']= df['Price EUR/mt'].shift(4)
df['Price_3']= df['Price EUR/mt'].shift(3)
df['Price_2']= df['Price EUR/mt'].shift(2)
df['Price_1']= df['Price EUR/mt'].shift(1)
df['Actual+7']= df['Price EUR/mt'].shift(-7)
df['Actual+15']=df['Price EUR/mt'].shift(-15)
df['Actual+30']=df['Price EUR/mt'].shift(-30)


df = df.iloc[30:-30]
st.write('\n\n\nInput File: \n\n\n')
st.write(df)
df.to_csv('Filtered_file.csv')
df.to_excel('Filtered_file.xlsx')


df=pd.read_csv('Filtered_file.csv')
X = df[['Price EUR/mt','Week of month','Month','Quarter','Price_7', 'Price_6', 'Price_5', 'Price_4','Price_3','Price_2','Price_1']]

y = df[['Actual+7','Actual+15','Actual+30']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



reg = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
reg.fit(X_train, y_train)

# Make predictions
y_pred = reg.predict(X_test)

y_full_pred=reg.predict(X)





df[['predicted+7','predicted+15','predicted+30']]=y_full_pred
st.write('\n\n\nFull data with Predicted and Actual values:\n\n\n')
st.write(df)
df.to_csv('FULL PREDICTED.csv')



y_pred_df = pd.DataFrame(y_pred, columns=[f'Predicted+7','predicted+15','predicted+30'])  




test_df_full = pd.concat([df[['Date']].tail(286).reset_index(drop=True),X_test.reset_index(drop=True), y_test.reset_index(drop=True),y_pred_df.reset_index(drop=True)], axis=1)

test_df_full.to_csv('Test_data_predicted.csv')
st.write('\n\n\nTest data with predicted and Actual:\n\n\n')
st.write(test_df_full)

###----------------------------------------------------------------------------------------------###

df=pd.read_csv("Gas and Oil Prices.csv")
unique_B = df.groupby('Argent Gas/Oil Category')['Argent Gas/Oil Price Name'].unique().reset_index()

# Create a new DataFrame for the categorized data
categorized_data = []

for index, row in unique_B.iterrows():
    category = row['Argent Gas/Oil Category']
    values = row['Argent Gas/Oil Price Name']
    
    for value in values:
        categorized_data.append([category, value])

categorized_df = pd.DataFrame(categorized_data, columns=['Intra Category', 'InterCategory'])
#st.write(df)
st.write(categorized_df)
# Save the categorized data to a new CSV file
categorized_df.to_csv('categorized_data.csv', index=False)



###----------------------------------------------------------------------------------------------###





import csv

# Define the input and output file names
input_file = 'Gas and Oil Prices.csv'
output_file = 'output.csv'

# Initialize a dictionary to store the matrix data
matrix_data = {}

# Read the input CSV file and populate the matrix data
with open(input_file, mode='r') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        
        row_heading = row["﻿Date"]
        col_heading = row['Argent Gas/Oil Price Name']
        value = row['Price EUR/mt']  # Assuming there is a 'Value' column for the matrix values
        if row_heading not in matrix_data:
            matrix_data[row_heading] = {}
        matrix_data[row_heading][col_heading] = value

# Get unique column headings from the 'B' column
column_headings = set()
for row in matrix_data.values():
    column_headings.update(row.keys())
column_headings = sorted(column_headings)

# Write the output CSV file
with open(output_file, mode='w', newline='') as outfile:
    writer = csv.writer(outfile)
    
    # Write the header row
    writer.writerow(['Date'] + column_headings)
    
    # Write the matrix data rows
    for row_heading, cols in matrix_data.items():
        row = [row_heading] + [cols.get(col, '') for col in column_headings]
        writer.writerow(row)
matrix_df=pd.read_csv('output.csv')



new_column_names = {
    'Unnamed: 1': 'SBO fob India'
}

# Rename the unnamed columns
matrix_df.rename(columns=new_column_names, inplace=True)

# Display the updated DataFrame

st.write(matrix_df)
matrix_df.to_csv('output.csv')
matrix = matrix_df.drop(columns='Date').corr()
st.write("Correlation matrix is : ")
st.write(matrix)







# Create a heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(matrix, annot=True, fmt=".2f", cmap='Blues')
2
# Save the plot as a jpg file
plt.savefig("correlation_matrix.jpg")
st.image("correlation_matrix.jpg")


source_file=pd.read_csv('FULL PREDICTED.csv')
Prompt = '''
\nYou are AI Assisstant.Read the given sales prediction data with actual data in given csv file{source_file} and take help from given correlation matrix{matrix_file}.
\nAnd answer the user query{user_query} with approppriate answer.Don't give hallunsinate response.
'''
prompt_template = PromptTemplate(template = Prompt ,input_variables = ["source_file","user_query","matrix_file"])

llm_chain=create_llm_chain_argent(prompt_template,0.5,3000)
if prompt := st.chat_input("Prompt...."):
  
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):

        st.markdown(llm_chain.run(source_file=source_file,user_query=prompt,matrix_file=matrix))
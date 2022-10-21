
import dowhy
from dowhy import CausalModel

import pandas as pd

#Parse raw data
raw_df = pd.read_csv('adult.data', header=None)
raw_df = raw_df.dropna()

#Select desired columns
input_data = { 'education': raw_df.iloc[:, 3],
	       'occupation': raw_df.iloc[:, 6],
	       'income': raw_df.iloc[:, 14]
}

df = pd.DataFrame(input_data)

#Make occupation column binary (Default or Exec-managerial)
#to create treatment 
for index, row in df.iterrows():
	if row['occupation'] != " Exec-managerial":
		row['occupation'] = "Default"

#Convert relevant categorical variables into indicators
df = pd.get_dummies(df, columns=['income', 'occupation'], drop_first=True)
df.columns=['education', 'income', 'occupation']

print(df.head())

###################################
#	      OUTPUT
###################################

#     education  income  occupation
# 0   Bachelors       0           1
# 1   Bachelors       0           0
# 2     HS-grad       0           1
# 3        11th       0           1
# 4   Bachelors       0           1

graph_1 = """	
	graph [
		directed 1
		
		node [id "education" label "education"]
		node [id "occupation" label "occupation"]
		node [id "income" label "income"]
		
		edge [source "education" target "occupation"]
		edge [source "education" target "income"]
		edge [source "occupation" target "income"]
	]
"""

#Create causal graph model
model_1 = CausalModel(
	data=df,
	treatment='occupation',
	outcome='income',
	graph=graph_1
)

#Identify estimand
estimand_1 = model_1.identify_effect()
print(estimand_1)

#Calculate estimation
estimate_1 = model_1.estimate_effect(
	identified_estimand=estimand_1,
	method_name='backdoor.linear_regression'
)

print(f'Estimate of causal effect: {estimate_1.value}')

#Validate results
refute1_results = model_1.refute_estimate(estimand_1, estimate_1,
			method_name="random_common_cause")			
print(refute1_results)

refute2_results = model_1.refute_estimate(estimand_1, estimate_1,
			method_name="placebo_treatment_refuter")			
print(refute2_results)












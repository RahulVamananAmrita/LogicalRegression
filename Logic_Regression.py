import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score , mean_absolute_error, mean_squared_error

df = pd.read_csv('Classification.csv')
print (df)

# manually add intercept
#df['intercept'] = 1
independent_variables = ['Hours_Studied']#, 'intercept']
x = df[independent_variables] # independent variable
y = df['Result'] # dependent variable
# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
model = model.fit(x, y)
 #check the accuracy on the training set
model.score(x, y)
# predict will give convert the probability(y=1) values > .5 to 1 else 0
print (model.predict(x)) 
print (model.predict_proba(x)[:,0])# predict_proba will return array containing probability of y = 0 and y = 1
### plotting fitted line
plt.scatter(df.Hours_Studied, y, color='black')
plt.yticks([0.0, 0.5, 1.0])
plt.plot(df.Hours_Studied, model.predict_proba(x)[:,1], color='blue',linewidth=3)
plt.title('Hours Studied vs Result')
plt.ylabel('Result')
plt.xlabel('Hours_Studied')
plt.show()
   





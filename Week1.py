
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('darkgrid')



df = pd.read_csv("box_office_predictions.csv")

df.describe()

# Remove films with "0" budget
df = df.loc[df.budget > 0,:]
df = df.dropna()

df["profit"] = df["gross"]-df["budget"]
df["year"] = df["name"].str[-5:-1].astype(int)


# Extreme Values trimming

billion = df['profit']>100000000
billion_l = df['profit']<-100000000
df["profit_tr"] = df['profit']

# Modify values in 'profit_tr' based on the masks

df.loc[billion, "profit_tr"] = 100000000
df.loc[billion_l, "profit_tr"] = -100000000

studio_counts = df.studio.value_counts()

# Tiers for sparser studios
three_timers = studio_counts[(studio_counts <= 3)].index
five_timers = studio_counts[(studio_counts > 3) & (studio_counts <= 5)].index
ten_timers = studio_counts[(studio_counts > 5) & (studio_counts <= 10)].index

# Combine sparse studios
df['studio'].replace(three_timers, 'Three Timer', inplace=True)
df['studio'].replace(five_timers, 'Five Timer', inplace=True)
df['studio'].replace(ten_timers, 'Ten Timer', inplace=True)

# Number of films from each star
star_counts = df.star.value_counts()

# Tiers for sparser stars
three_timers = star_counts[(star_counts <= 3)].index
five_timers = star_counts[(star_counts > 3) & (star_counts <= 5)].index
ten_timers = star_counts[(star_counts > 5) & (star_counts <= 10)].index

# Combine sparse stars
df['star'].replace(three_timers, 'Three Timer', inplace=True)
df['star'].replace(five_timers, 'Five Timer', inplace=True)
df['star'].replace(ten_timers, 'Ten Timer', inplace=True)

# Number of films for each director
director_counts = df.director.value_counts()

#Tier for sparser directors
three_timers = director_counts[director_counts<=3].index
five_timers = director_counts[(director_counts>3)&(director_counts<=5)].index
ten_timers = director_counts[(director_counts>5)&(director_counts<=10)].index

df['director'].replace(three_timers, 'Three Timer', inplace = True)
df['director'].replace(five_timers, 'Five Timer', inplace = True)
df['director'].replace(ten_timers, 'Ten Timer', inplace = True)

# Number of films from each country
country_counts = df.country.value_counts()

# Combine countries with fewer than 50 films
other_countries = country_counts[country_counts < 50].index
df['country'].replace(other_countries, 'Other', inplace=True)

# Number of films in each genre
genre_counts = df.genre.value_counts()

# Combine genres with fewer than 50 films
other_genres = genre_counts[genre_counts < 100].index
df['genre'].replace(other_genres, 'Other', inplace=True)

# New genre frequencies
df.genre.value_counts()

# Fix "unrated" labels
df['rating'].replace(['NOT RATED', 'UNRATED', 'Not specified'], 'NR', inplace=True)




df[['score', 'votes', 'profit', 'budget']].corr()






# Create analytical base table (ABT)
abt = pd.get_dummies ( df.drop(['name', 'gross', 'votes', 'score'], axis=1) )





train = abt[abt.year < 2014]
test = abt[abt.year >= 2015]




# Group by 'studio' and compute the mean for 'budget'
avg_budget_per_studio = df.groupby('studio')['budget'].mean().sort_values(ascending=True)

# Plot
avg_budget_per_studio.plot(kind='bar', figsize=(15,7))
plt.title('Average Budget per Studio')
plt.ylabel('Average Budget')
plt.xlabel('Studio')
plt.show()

print("From this figure, it is clear that budget and studios are correlated, so I would like to drop budget from X")




y_train = train.profit_tr
X_train = train.drop(['profit_tr','profit','budget'], axis=1)

y_test = test.profit_tr
X_test = test.drop(['profit_tr','profit', 'budget'], axis=1)





rf = RandomForestRegressor(n_estimators=500,random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
y_test = test["profit"]




sns.scatterplot(y_test, pred)
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')
plt.show()



r2_score(y_test, pred)




# For this analysis, we would like to determine the optimal n_estimator.
def finetune_r(X_train, y_train, X_test, y_test):
    r2 = []
    for i in [5, 10, 30, 50, 100, 150, 300, 500]:
        rf = RandomForestRegressor(n_estimators = i, random_state = 42)
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        r2.append(r2_score(y_test, pred))
    return r2
    
    
r2_est = finetune_r(X_train, y_train, X_test, y_test)




estimators = [5, 10, 30, 50, 100, 150, 300, 500]

plt.plot(r2_est, marker='o')
plt.xlabel("Number of Estimators")
plt.ylabel("R^2 Score")
plt.xticks(ticks=range(len(r2_est)), labels=["5", "10", "30", "50", "100", "150", "300", "500"])
plt.title("R^2 Score vs. Number of Estimators")
plt.grid(True)
plt.show()

print("From the figure above, we can conclude that 50 is optimal considering R^2" )



# For this analysis, we would like to determine the optimal n_estimator considering MSE instead. 
def finetune_m(X_train, y_train, X_test, y_test):
    mse = []
    for i in [5, 10, 30, 50, 100, 150, 300, 500]:
        rf = RandomForestRegressor(n_estimators = i, random_state = 42)
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        mse.append(mean_squared_error(y_test, pred))
    return mse
    
    
mse_est = finetune_m(X_train, y_train, X_test, y_test)




estimators = [5, 10, 30, 50, 100, 150, 300, 500]

plt.plot(mse_est, marker='o')
plt.xlabel("Number of Estimators")
plt.ylabel("MSE")
plt.xticks(ticks=range(len(r2_est)), labels=["5", "10", "30", "50", "100", "150", "300", "500"])
plt.title("MSE vs. Number of Estimators")
plt.grid(True)
plt.show()

print("From the figure above, we can conclude that 50 is optimal considering MSE" )




rf = RandomForestRegressor(n_estimators=50,random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
y_test = test["profit"]

mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")



# Create new analytical base table (ABT)
abt_ps = pd.get_dummies ( df.drop(['name', 'gross', 'votes'], axis=1) )



train = abt_ps[abt_ps.year <= 2014]
test = abt_ps[abt_ps.year > 2015]

y_train = train.profit_tr
X_train = train.drop(['profit_tr', 'profit', 'budget'], axis=1)

y_test = test.profit
X_test = test.drop(['profit', 'profit_tr', 'budget'], axis=1)




# Train a basic random forest model
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Make prediction on test set
pred = rf.predict(X_test)




sns.scatterplot(y_test, pred)
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')
plt.show()



estimators = [5, 10, 30, 50, 100, 150, 300, 500]

r2_est = finetune_r(X_train, y_train, X_test, y_test)

plt.plot(r2_est, marker='o')
plt.xlabel("Number of Estimators")
plt.ylabel("R^2 Score")
plt.xticks(ticks=range(len(r2_est)), labels=["5", "10", "30", "50", "100", "150", "300", "500"])
plt.title("R^2 Score vs. Number of Estimators")
plt.grid(True)
plt.show()

print("From the figure above, we can conclude that 150 is optimal considering R^2" )




estimators = [5, 10, 30, 50, 100, 150, 300, 500]

mse_est = finetune_m(X_train, y_train, X_test, y_test)

plt.plot(mse_est, marker='o')
plt.xlabel("Number of Estimators")
plt.ylabel("MSE")
plt.xticks(ticks=range(len(r2_est)), labels=["5", "10", "30", "50", "100", "150", "300", "500"])
plt.title("MSE vs. Number of Estimators")
plt.grid(True)
plt.show()

print("From the figure above, we can conclude that 150 is optimal considering MSE" )


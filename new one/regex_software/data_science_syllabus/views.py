from django.shortcuts import render
import sklearn 

# Create your views here.
import pandas as pd

df = pd.read_csv("covid_toy.csv")
print(df.head())
# from sklearn.preprocessing import LabelEncoder
# lb = LabelEncoder()
# # df['gender'] = lb.fit_transform(df['gender'])
# # df['cough'] = lb.fit_transform(df['cough'])
# # df['has_covid'] = lb.fit_transform(df['has_covid'])
# # df['city'] = lb.fit_transform(df['city'])
# print("Before label encoding:\n", df.head())

# df['gender'] = lb.fit_transform(df['gender'])
# print("After encoding 'gender':\n", df.head())
# df['cough'] = lb.fit_transform(df['cough'])
# print("After encoding 'cough':\n", df.head())
# df['city'] = lb.fit_transform(df['city'])
# print("After encoding 'city':\n", df.head())
# df['has_covid'] = lb.fit_transform(df['has_covid'])
# print("After encoding 'has_covid':\n", df.head())

# Repeat for other columns...

# x = df.iloc[:, 2:4].values
# y = df.iloc[:, -1].values
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=5)
# knn_ans = knn.fit(x_train, y_train)
# # print(x_train.head())

# print(df.head())

from sklearn.preprocessing import LabelEncoder 

lb = LabelEncoder() 
df['gener'] = lb.fit_transform(df['gender']) 
df['cough'] = lb.fit_transform(df['cough']) 
df['city'] = lb.fit_transform(df['city']) 
df['has_covid'] = lb.fit_transform(df['has_covid'])

print(df.head())  


df.head()

def home(request):
    return render(request,"index.html")

def predict(request):
    if request.method == 'POST':
        a = request.POST.get('age')
        age = int(a)
        s = request.POST.get('cough')
        salary = int(s)
        result = knn_ans.predict([[age,cough]])[0]

print(df.head())
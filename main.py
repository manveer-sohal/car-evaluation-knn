import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import  preprocessing




#reads in data from student-mat.csv usings pandas, data is seperated by ';'
data = pd.read_csv("car.data", sep = ",")

#takes the labels and encodes them into appropriate integer values
le = preprocessing.LabelEncoder()

#gets the entire coloum of each these labels, makes it into a list, and turns them into integer values
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))


predict = "class"

X = list(zip(buying,maint,door,persons,lug_boot,safety))

Y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train,y_train)
acc= model.score(x_test,y_test)

predicted = model.predict(x_test)

names = ["unacc" , "acc", "good", "vgood"]


print("Model Performance Summary:\n---------------------------")
print("R^2 Score: ",acc)

print("\nModel Data:\n---------------------------")

for x in range(len(predicted)):

    print("predicted:", names[predicted[x]], "Data" , x_test[x], "actual:" , names[y_test[x]])
    pass



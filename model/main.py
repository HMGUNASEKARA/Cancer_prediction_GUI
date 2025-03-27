import pandas as pd 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
import pickle as pickle


# This function data importing and cleaninig part
def get_clean_data():
    data = pd.read_csv("data/breast_cancer.csv")
    #print(data.head())
    data = data.drop(['Unnamed: 32','id'],axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
    return data  # return the function value in to data 

# This function build the model  
def create_model(data):
    x = data.drop(['diagnosis'],axis=1)
    y = data['diagnosis']

    # scale the data 
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    #spit the data 
    x_train,x_test,y_train,y_test = train_test_split(
        x,y,test_size=0.2,random_state= 42
    )

    # train model 
    model =LogisticRegression()
    model.fit(x_train,y_train)

    # test model 
    y_pred = model.predict(x_test)
    print('Accuracy of this model: ',accuracy_score(y_test,y_pred))
    print('Classification report \n',classification_report(y_test,y_pred))


    return model,scaler



def main():
    data =get_clean_data() 
    
    model,scaler  = create_model(data)

# by using pickle5 package we convart this modeland scale in to binary file
# Then we can use this by using 
    
    with open('model/model.pkl','wb') as f:
        pickle.dump(model, f)
    
    with open('model/scaler.pkl','wb') as f:
        pickle.dump(scaler, f)

if __name__ == '__main__':
    main()
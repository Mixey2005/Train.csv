import pandas as pd

df = pd.read_csv('train.csv')
df.drop(['id','bdate', 'has_photo', 'has_mobile', 'followers_count', 'graduation', 'last_seen','career_start', 'career_end','city','life_main','people_main','occupation_type','occupation_name'], axis = 1, inplace = True)
df[list(pd.get_dummies(df['education_form']).columns)] =pd.get_dummies(df['education_form'])
df['education_form'].fillna('Full-time',inplace=True)
df.drop(['education_form'],axis=1, inplace=True)


def edu_status_apply(edu_status):
    if edu_status == 'Undergraduate applicant':
        return 0
    elif edu_status == 'Student (Specialist)' or edu_status == "Student (Bachelor's)" or edu_status == "Student (Master's)" :
        return 1
    elif edu_status == 'Alumnus (Specialist)' or edu_status == "Alumnus (Bachelor's)" or edu_status == "Alumnus (Master's)":
        return 2
    elif edu_status == 'PhD' or edu_status == 'Candidate of Sciences':
        return 3

def split_langs(langs):
    return langs.split(';')
df['langs'] = df['langs'].apply(split_langs)
df['langs'] = df['langs'].apply(len)
print(df['langs'].value_counts())



df['education_status'] = df['education_status'].apply(edu_status_apply)
print(df['education_status'].value_counts())

df.info()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score

X = df.drop('result',axis=1)
y = df['result']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
print('Процент правильно предсказанных исходов:', round(accuracy_score(y_test,y_pred) * 100 ,2))



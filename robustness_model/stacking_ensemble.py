from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

X_train, X_val, Y_train, Y_val = train_test_split(clean_df.drop('Dyslexic', axis=1), clean_df['Dyslexic'], test_size=0.3, random_state=2)

base_classifiers = [
    ('dt', DecisionTreeClassifier()),
    ('rf', RandomForestClassifier()),
    ('svc', SVC())
]

meta_learner = LogisticRegression()

stacked_classifier = StackingClassifier(estimators=base_classifiers, final_estimator=meta_learner)

stacked_classifier.fit(X_train, Y_train)

performance_eval(stacked_classifier, X_val, Y_val)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

y = [0.953, 0.987, 0.981, 0.964, 0.978, 0.956, 0.988, 0.933, 0.968]
x = ['Adaboost', 'Decisison Tree', 'KNN-Classifier', 'LDA-Classifier', 'Logistic Regression', 'MLP-Classifier', 'Naive Bayes', 'Random Forest', 'SVM']

plt.bar(x, y, color ='maroon', width = 0.4)
ax.plot(x,y)
plt.show()
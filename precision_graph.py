import matplotlib.pyplot as plt

fig, ax = plt.subplots()

y = [0.967, 0.956, 0.978, 0.948, 0.932, 0.953, 0.988, 0.975, 0.979]
x = ['Adaboost', 'Decisison Tree', 'KNN-Classifier', 'LDA-Classifier', 'Logistic Regression', 'MLP-Classifier', 'Naive Bayes', 'Random Forest', 'SVM']

plt.bar(x, y, color ='blue', width = 0.4)
ax.plot(x,y)
plt.show()
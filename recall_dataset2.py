import matplotlib.pyplot as plt

fig, ax = plt.subplots()

y = [0.964, 0.972, 0.935, 0.973, 0.948, 0.966, 0.979, 0.931, 0.953]
x = ['Adaboost', 'Decisison Tree', 'KNN-Classifier', 'LDA-Classifier', 'Logistic Regression', 'MLP-Classifier', 'Naive Bayes', 'Random Forest', 'SVM']

plt.bar(x, y, color ='blue', width = 0.4)
ax.plot(x,y)
plt.show()
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

y = [0.981, 0.977, 0.965, 0.943, 0.988, 0.961, 0.978, 0.955, 0.985]
x = ['Adaboost', 'Decisison Tree', 'KNN-Classifier', 'LDA-Classifier', 'Logistic Regression', 'MLP-Classifier', 'Naive Bayes', 'Random Forest', 'SVM']

plt.bar(x, y, color ='maroon', width = 0.4)
ax.plot(x,y)
plt.show()
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

y = [0.971, 0.967, 0.975, 0.950, 0.946, 0.968, 0.978, 0.965, 0.984]
x = ['Adaboost', 'Decisison Tree', 'KNN-Classifier', 'LDA-Classifier', 'Logistic Regression', 'MLP-Classifier', 'Naive Bayes', 'Random Forest', 'SVM']

plt.bar(x, y, color ='blue', width = 0.4)
ax.plot(x,y)
plt.show()
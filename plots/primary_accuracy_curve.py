import matplotlib.pyplot as plt

# Sample accuracy values and algorithm labels (replace with your actual data)
accuracy_values = [0.931, 0.946, 0.913, 0.928, 0.935, 0.926, 0.958, 0.933, 0.929, 0.937]
algorithm_labels = ['Adaboost', 'Decision Tree', 'KNN', 'LDA', 'Logistic', 'MLP', 'Naive Bayes', 'Random Forest', 'SVM', 'Ensemble']

# Create a list of x-axis values (e.g., epochs or iterations)
epochs = range(1, len(accuracy_values) + 1)

# Create the accuracy curve
plt.figure(figsize=(4, 4))
plt.plot(epochs, accuracy_values, marker='o', linestyle='-', color='b')
plt.title('Accuracy Curve')
plt.xlabel('Epochs/Iterations')
plt.ylabel('Accuracy')
plt.grid(True)

# Add algorithm labels on the x-axis
plt.xticks(epochs, algorithm_labels, rotation=45)

# Save the curve as a PNG file
plt.savefig('primary_accuracy_curve.png')

# Display the plot (optional)
plt.show()

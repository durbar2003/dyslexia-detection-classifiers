import matplotlib.pyplot as plt

# Sample accuracy values and algorithm labels (replace with your actual data)
accuracy_values = [0.942, 0.931, 0.917, 0.928, 0.914, 0.926, 0.959, 0.947, 0.933, 0.926]
algorithm_labels = ['Adaboost', 'Decision Tree', 'KNN', 'LDA', 'Logistic', 'MLP', 'Naive Bayes', 'Random Forest', 'SVM', 'Ensemble']

# Create a list of x-axis values (e.g., epochs or iterations)
epochs = range(1, len(accuracy_values) + 1)

# Create the accuracy curve with a smaller size
plt.figure(figsize=(4, 4))  # Adjust the figsize for a smaller plot
plt.plot(epochs, accuracy_values, marker='o', linestyle='-', color='b')
plt.title('Accuracy Curve')
plt.ylabel('Accuracy')
plt.grid(True)

# Add algorithm labels on the x-axis
plt.xticks(epochs, algorithm_labels, rotation=45)

# Save the smaller curve as a PNG file
plt.savefig('robustness_accuracy_curve_smaller.png')

# Display the smaller plot (optional)
plt.show()

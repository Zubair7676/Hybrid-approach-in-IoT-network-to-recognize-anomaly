correlation_matrix = train[num_col].corr()


plt.figure(figsize=(12, 12))


sns.heatmap(correlation_matrix, cmap='coolwarm', square=True)

# Add labels and title
plt.xlabel('Features')
plt.ylabel('Features')
plt.title('Correlation Heatmap of Features')

# Rotate x-axis labels for better readability with many features
plt.xticks(rotation=90)
plt.yticks(rotation=0)

# Show the plot
plt.show()
train
# labels = train['Label']

# # Create a count plot with Seaborn
# sns.countplot(x=labels)

# # Add labels and title
# plt.xlabel('Class Label')
# plt.ylabel('Number of Data Points')
# plt.title('Class Distribution')

# # Rotate x-axis labels for better readability if there are many classes
# plt.xticks(rotation=0)

# # Show the plot
# plt.show()
labels = train['attack_cat']

# Create a count plot with Seaborn
sns.countplot(x=labels)

# Add labels and title
plt.xlabel('Class Label')
plt.ylabel('Number of Data Points')
plt.title('Class Distribution')

# Rotate x-axis labels for better readability if there are many classes
plt.xticks(rotation=90)

# Show the plot
plt.show()
# combined_data.drop(columns=['attack_cat'],inplace=True)
del combined_data
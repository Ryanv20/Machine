from sklearn.preprocessing import LabelEncoder

# List of categorical strings
categories = ["High", "Medium", "Low", "High", "Low"]

# Use LabelEncoder to encode them into numbers
label_encoder = LabelEncoder()
encoded_categories = label_encoder.fit_transform(categories)

print(encoded_categories)  # Output: [0, 1, 2, 0, 2]


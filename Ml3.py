import tensorflow as tf

# Create a constant tensor
tensor = tf.constant([[1, 2], [3, 4]])
print("Tensor:")
print(tensor)

# Perform basic operations
tensor_add = tensor + 5
print("\nTensor after adding 5:")
print(tensor_add)

# Create a variable tensor
var = tf.Variable([[1.0, 2.0], [3.0, 4.0]])
print("\nVariable Tensor:")
print(var)

# Update the variable
var.assign_add([[1.0, 1.0], [1.0, 1.0]])
print("\nUpdated Variable Tensor:")
print(var)


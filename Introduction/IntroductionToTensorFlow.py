from numpy import number
import tensorflow as tf

print(tf.version)

'''
--> Tensors

"A tensor is a generalization of vectors and matrices to potentially higher dimensions. Internally, TensorFlow represents tensors as n-dimensional arrays of base datatypes." (https://www.tensorflow.org/guide/tensor)
It should't surprise you that tensors are a fundemental apsect of TensorFlow. They are the main objects that are passed around and manipluated throughout the program.
Each tensor represents a partialy defined computation that will eventually produce a value. TensorFlow programs work by building a graph of Tensor objects that details how tensors are related.
Running different parts of the graph allow results to be generated.
Each tensor has a data type and a shape.
Data Types Include: float32, int32, string and others.
Shape: Represents the dimension of data.
Just like vectors and matrices tensors can have operations applied to them like addition, subtraction, dot product, cross product etc.
In the next sections we will discuss some different properties of tensors.
This is to make you more familiar with how tensorflow represnts data and how you can manipulate this data.

--> Creating Tensors

Below is an example of how to create some different tensors.
You simply define the value of the tensor and the datatype and you are good to go! It's worth mentioning that usually we deal with tensors of numeric data, it is quite rare to see string tensors.
For a full list of datatypes please refer to the following guide.
'''

# Rank is 0 for the below syntax because it can store multiple data in a list
string = tf.Variable("this is a string", tf.string)
number1 = tf.Variable(324, tf.int16)
number2 = tf.Variable(324, tf.int32)
number3 = tf.Variable(324, tf.int64)
float1 = tf.Variable(3.567, tf.float16)
float2 = tf.Variable(3.567, tf.float32)
float3 = tf.Variable(3.567, tf.float64)

print(string)
print(number1)
print(number2)
print(number3)
print(float1)
print(float2)
print(float3)
print(tf.rank(string)) # tf.Tensor(0, shape=(), dtype=int32)

'''
--> Rank/Degree of Tensors

Another word for rank is degree, these terms simply mean the number of dimensions involved in the tensor.
What we created above is a tensor of rank 0, also known as a scalar.
Now we'll create some tensors of higher degrees/ranks.
'''
# Rank is 1 (tf.Tensor(1, shape=(), dtype=int32)) for the below syntax because it can store multiple data in a list
rank1_tensor = tf.Variable(["Test"],tf.string)
print(tf.rank(rank1_tensor))

# Rank is 2 (tf.Tensor(2, shape=(), dtype=int32))for the below syntax because it can store multiple data in a list
rank2_tensor = tf.Variable([["Test1", "ok1"],["Test2","ok2"]],tf.string)
print(tf.rank(rank2_tensor))

tensor1 = tf.ones([1,2,3])
tensor2 = tf.reshape(tensor1, [2,3,1])
tensor3 = tf.reshape(tensor2, [3, -1])

print(tensor1)

print(tensor2)

print(tensor3)

# with tf.Session() as sees:
#     tensor.eval()
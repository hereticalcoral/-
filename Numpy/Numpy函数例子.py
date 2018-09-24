#例子来源: https://www.tutorialspoint.com/numpy

# minimum dimensions 
import numpy as np 
a = np.array([1, 2, 3,4,5], ndmin = 2) 
print a

##ndarray.shape  and reshape :This array attribute returns a tuple consisting of array dimensions. It can also be used to resize the array.
import numpy as np 
a = np.array([[1,2,3],[4,5,6]]) 
print a.shape

# this resizes the ndarray 
import numpy as np 
a = np.array([[1,2,3],[4,5,6]]) 
a.shape = (3,2) 
print a 

import numpy as np 
a = np.array([[1,2,3],[4,5,6]]) 
b = a.reshape(3,2) 
print b

##ndarray.ndim :This array attribute returns the number of array dimensions
# an array of evenly spaced numbers 
import numpy as np 
a = np.arange(24) 
print a

# this is one dimensional array 
import numpy as np 
a = np.arange(24) 
a.ndim  
# now reshape it 
b = a.reshape(2,4,3) 
print b 
# b is having three dimensions

##numpy.itemsize : This array attribute returns the length of each element of array in bytes.
# dtype of array is int8 (1 byte) 
import numpy as np 
x = np.array([1,2,3,4,5], dtype = np.int8) 
print x.itemsize

# dtype of array is now float32 (4 bytes) 
import numpy as np 
x = np.array([1,2,3,4,5], dtype = np.float32) 
print x.itemsize

##numpy.empty
#It creates an uninitialized array of specified shape and dtype. It uses the following constructor

import numpy as np 
x = np.empty([3,2], dtype = int) 
print x

##numpy.zeros
#Returns a new array of specified size, filled with zeros.
# array of five zeros. Default dtype is float 
import numpy as np 
x = np.zeros(5) 
print x

import numpy as np 
x = np.zeros((5,), dtype = np.int) 
print x

# custom type 
import numpy as np 
x = np.zeros((2,2), dtype = [('x', 'i4'), ('y', 'i4')])  
print x

##numpy.ones
Returns a new array of specified size and type, filled with ones.
# array of five ones. Default dtype is float 
import numpy as np 
x = np.ones(5) 
print x

import numpy as np 
x = np.ones([2,2], dtype = int) 
print x

##numpy.asarray
This function is similar to numpy.array except for the fact that it has fewer parameters. 
This routine is useful for converting Python sequence into ndarray.
# convert list to ndarray 

import numpy as np 
x = [1,2,3] 
a = np.asarray(x) 
print a

# dtype is set 
import numpy as np 
x = [1,2,3]
a = np.asarray(x, dtype = float)   #注意此处特意标明float，所以输出结果与前一个例子不太一样
print a

# ndarray from tuple 
import numpy as np 
x = (1,2,3) 
a = np.asarray(x) 
print a

# ndarray from list of tuples 
import numpy as np 
x = [(1,2,3),(4,5)] 
a = np.asarray(x) 
print a

##numpy.frombuffer
This function interprets a buffer as one-dimensional array. 
Any object that exposes the buffer interface is used as parameter to return an ndarray.
import numpy as np 
s = 'Hello World' 
a = np.frombuffer(s, dtype = 'S1') 
print a

##numpy.fromiter
This function builds an ndarray object from any iterable object.
A new one-dimensional array is returned by this function
# create list object using range function 
import numpy as np 
list = range(5) 
print list

# obtain iterator object from list 
import numpy as np 
list = range(5) 
it = iter(list)  
# use iterator to create ndarray 
x = np.fromiter(it, dtype = float) 
print x


##numpy.arange
This function returns an ndarray object containing evenly spaced values within a given range. 
The format of the function is as follows numpy.arange(start, stop, step, dtype)
import numpy as np 
x = np.arange(5) 
print x

import numpy as np 
# dtype set 
x = np.arange(5, dtype = float)
print x

# start and stop parameters set 
import numpy as np 
x = np.arange(10,20,2) 
print x


##numpy.linspace
This function is similar to arange() function. In this function, instead of step size, 
the number of evenly spaced values between the interval is specified. The usage of this function is as follows
numpy.linspace(start, stop, num, endpoint, retstep, dtype)
import numpy as np 
x = np.linspace(10,20,5) 
print x

# endpoint set to false 
import numpy as np 
x = np.linspace(10,20, 5, endpoint = False) 
print x

# find retstep value 
import numpy as np 

x = np.linspace(1,2,5, retstep = True) 
print x 
# retstep here is 0.25


##numpy.logspace
This function returns an ndarray object that contains the numbers that are evenly spaced on a log scale. 
Start and stop endpoints of the scale are indices of the base, usually 10.
numpy.logspace(start, stop, num, endpoint, base, dtype)

import numpy as np 
# default base is 10 
a = np.logspace(1.0, 2.0, num = 10) 
print a

# set base of log space to 2 
import numpy as np 
a = np.logspace(1,10,num = 10, base = 2) 
print a


##NumPy - Indexing & Slicing
Contents of ndarray object can be accessed and modified by indexing or slicing, just like Python's in-built container objects.

As mentioned earlier, items in ndarray object follows zero-based index. Three types of indexing methods are available − field access, 
basic slicing and advanced indexing. Basic slicing is an extension of Python's basic concept of slicing to n dimensions. 
A Python slice object is constructed by giving start, stop, and step parameters to the built-in slice function. 
This slice object is passed to the array to extract a part of array.
import numpy as np 
a = np.arange(10) 
s = slice(2,7,2) 
print a[s]

import numpy as np 
a = np.arange(10) 
b = a[2:7:2] 
print b

## Indexinh and Slicing
If only one parameter is put, a single item corresponding to the index will be returned. If a : is inserted in front of it, 
all items from that index onwards will be extracted. If two parameters (with : between them) is used, 
items between the two indexes (not including the stop index) with default step one are sliced
# slice single item 
import numpy as np 
a = np.arange(10) 
b = a[5] 
print b

# slice items starting from index 
import numpy as np 
a = np.arange(10) 
print a[2:]

# slice items between indexes 
import numpy as np 
a = np.arange(10) 
print a[2:5]

import numpy as np 
a = np.array([[1,2,3],[3,4,5],[4,5,6]]) 
print a  
# slice items starting from index
print 'Now we will slice the array from the index a[1:]' 
print a[1:]

# array to begin with 
import numpy as np 
a = np.array([[1,2,3],[3,4,5],[4,5,6]]) 
print 'Our array is:' 
print a 
print '\n'  
# this returns array of items in the second column 
print 'The items in the second column are:'  
print a[...,1] 
print '\n'  
# Now we will slice all items from the second row 
print 'The items in the second row are:' 
print a[1,...] 
print '\n'  
# Now we will slice all items from column 1 onwards 
print 'The items column 1 onwards are:' 
print a[...,1:]


##Advanced Indexing    https://www.tutorialspoint.com/numpy/numpy_advanced_indexing.htm
import numpy as np 
x = np.array([[1, 2], [3, 4], [5, 6]]) 
y = x[[0,1,2], [0,1,0]] 
print y

import numpy as np    ?
x = np.array([[ 0,  1,  2],[ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]]) 
print 'Our array is:' 
print x 
print '\n' 
rows = np.array([[0,0],[3,3]])
cols = np.array([[0,2],[0,2]]) 
y = x[rows,cols] 
print 'The corner elements of this array are:' 
print y

import numpy as np 
x = np.array([[ 0,  1,  2],[ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]]) 
print 'Our array is:' 
print x 
print '\n'  
# slicing 
z = x[1:4,1:3] 
print 'After slicing, our array becomes:' 
print z 
print '\n'  
# using advanced index for column 
y = x[1:4,[1,2]] 
print 'Slicing using advanced index for column:' 
print y


##Boolean Array Indexing
This type of advanced indexing is used when the resultant object is meant to be the result of Boolean operations, such as 
comparison operators

import numpy as np 
x = np.array([[ 0,  1,  2],[ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]]) 
print 'Our array is:' 
print x 
print '\n'  
# Now we will print the items greater than 5 
print 'The items greater than 5 are:' 
print x[x > 5]

import numpy as np 
a = np.array([np.nan, 1,2,np.nan,3,4,5]) 
print a[~np.isnan(a)]     #NaN (Not a Number) elements are omitted by using ~ (complement operator).

import numpy as np 
a = np.array([1, 2+6j, 5, 3.5+5j]) 
print a[np.iscomplex(a)]


##Broadcasting
import numpy as np 
a = np.array([1,2,3,4]) 
b = np.array([10,20,30,40]) 
c = a * b 
print c

import numpy as np ?
a = np.array([[0.0,0.0,0.0],[10.0,10.0,10.0],[20.0,20.0,20.0],[30.0,30.0,30.0]]) 
b = np.array([1.0,2.0,3.0])  
print 'First array:' 
print a 
print '\n'  
print 'Second array:' 
print b 
print '\n'  
print 'First Array + Second Array' 
print a + b


##Iterating Over Array  https://www.tutorialspoint.com/numpy/numpy_iterating_over_array.htm
import numpy as np
a = np.arange(0,60,5)
a = a.reshape(3,4)
print 'Original array is:'
print a
print '\n'
print 'Modified array is:'
for x in np.nditer(a):
   print x,

import numpy as np 
a = np.arange(0,60,5) 
a = a.reshape(3,4) 
print 'Original array is:'
print a 
print '\n'     
print 'Transpose of the original array is:' 
b = a.T 
print b 
print '\n'     
print 'Modified array is:' 
for x in np.nditer(b): 
   print x,

import numpy as np
a = np.arange(0,60,5)
a = a.reshape(3,4)
print 'Original array is:'
print a
print '\n'
print 'Transpose of the original array is:'
b = a.T
print b
print '\n'
print 'Sorted in C-style order:'
c = b.copy(order = 'C')
print c
for x in np.nditer(c):
   print x,
print '\n'
print 'Sorted in F-style order:'
c = b.copy(order = 'F')
print c
for x in np.nditer(c):
   print x,
   
import numpy as np 
a = np.arange(0,60,5) 
a = a.reshape(3,4) 
print 'Original array is:' 
print a 
print '\n'  
print 'Sorted in C-style order:' 
for x in np.nditer(a, order = 'C'): 
   print x,  
print '\n' 
print 'Sorted in F-style order:' 
for x in np.nditer(a, order = 'F'): 
   print x,
   

##Modifying Array Values
import numpy as np
a = np.arange(0,60,5)
a = a.reshape(3,4)
print 'Original array is:'
print a
print '\n'
for x in np.nditer(a, op_flags = ['readwrite']):
   x[...] = 2*x
print 'Modified array is:'
print a
   
import numpy as np 
a = np.arange(0,60,5) 
a = a.reshape(3,4) 
print 'Original array is:' 
print a 
print '\n'  
print 'Modified array is:' 
for x in np.nditer(a, flags = ['external_loop'], order = 'F'):
   print x,   


##Broadcasting Iteration
If two arrays are broadcastable, a combined nditer object is able to iterate upon them concurrently. Assuming that an array a has 
dimension 3X4, and there is another array b of dimension 1X4, the iterator of following type is used (array b is broadcast to size of a).
import numpy as np 
a = np.arange(0,60,5) 
a = a.reshape(3,4) 
print 'First array is:' 
print a 
print '\n'  
print 'Second array is:' 
b = np.array([1, 2, 3, 4], dtype = int) 
print b  
print '\n' 
print 'Modified array is:' 
for x,y in np.nditer([a,b]): 
   print "%d:%d" % (x,y),


##Array Manipulation https://www.tutorialspoint.com/numpy/numpy_array_manipulation.htm
They can be categorized into these kinds:
Changing Shape: reshape, flat, flatten, ravel
Transpose Operations: transpose, ndarray.T, rollaxis, swapaxes
Changing Dimensions: broadcast, broadcast_to, expand_dimss, squeeze
Joining Arrays: concatenate, stack, hstack, vstack
Splitting Arrays: split, hsplit, vsplit
Adding/Removing Elements: resize, append, insert, delete, unique


##Binary Operators https://www.tutorialspoint.com/numpy/numpy_binary_operators.htm
bitwise_and, bitwise_or, invert, left_shift, right_shift


##Strings Functions  https://www.tutorialspoint.com/numpy/numpy_string_functions.htmf
The following functions are used to perform vectorized string operations for arrays of dtype numpy.string_ or numpy.unicode_. 
They are based on the standard string functions in Python's built-in library.
add, multiply, center, capitalize, title, lower, upper, split, splitlines, strip, join, replace, decode, encode


##Mathematical Functions https://www.tutorialspoint.com/numpy/numpy_mathematical_functions.htm
##Trigonometric Functions
import numpy as np 
a = np.array([0,30,45,60,90]) 
print 'Sine of different angles:' 
# Convert to radians by multiplying with pi/180 
print np.sin(a*np.pi/180) 
print '\n'  
print 'Cosine values for angles in array:' 
print np.cos(a*np.pi/180) 
print '\n'  
print 'Tangent values for given angles:' 
print np.tan(a*np.pi/180) 

import numpy as np 
a = np.array([0,30,45,60,90]) 

print 'Array containing sine values:' 
sin = np.sin(a*np.pi/180) 
print sin 
print '\n'  

print 'Compute sine inverse of angles. Returned values are in radians.' 
inv = np.arcsin(sin) 
print inv 
print '\n'  

print 'Check result by converting to degrees:' 
print np.degrees(inv) 
print '\n'  

print 'arccos and arctan functions behave similarly:' 
cos = np.cos(a*np.pi/180) 
print cos 
print '\n'  

print 'Inverse of cos:' 
inv = np.arccos(cos) 
print inv 
print '\n'  

print 'In degrees:' 
print np.degrees(inv) 
print '\n'  

print 'Tan function:' 
tan = np.tan(a*np.pi/180) 
print tan
print '\n'  

print 'Inverse of tan:' 
inv = np.arctan(tan) 
print inv 
print '\n'  

print 'In degrees:' 
print np.degrees(inv)

##Functions for Rounding
numpy.around(a,decimals)

import numpy as np 
a = np.array([1.0,5.55, 123, 0.567, 25.532]) 
print 'Original array:' 
print a 
print '\n'  
print 'After rounding:' 
print np.around(a) 
print np.around(a, decimals = 1) 
print np.around(a, decimals = -1)

numpy.floor()
import numpy as np 
a = np.array([-1.7, 1.5, -0.2, 0.6, 10]) 
print 'The given array:' 
print a 
print '\n'  
print 'The modified array:' 
print np.floor(a)

numpy.ceil()
import numpy as np 
a = np.array([-1.7, 1.5, -0.2, 0.6, 10]) 
print 'The given array:' 
print a 
print '\n'  
print 'The modified array:' 
print np.ceil(a)


###Arithemtic Operations https://www.tutorialspoint.com/numpy/numpy_arithmetic_operations.htm
import numpy as np 
a = np.arange(9, dtype = np.float_).reshape(3,3) 

print 'First array:' 
print a 
print '\n'  

print 'Second array:' 
b = np.array([10,10,10]) 
print b 
print '\n'  

print 'Add the two arrays:' 
print np.add(a,b) 
print '\n'  

print 'Subtract the two arrays:' 
print np.subtract(a,b) 
print '\n'  

print 'Multiply the two arrays:' 
print np.multiply(a,b) 
print '\n'  

print 'Divide the two arrays:' 
print np.divide(a,b)

##numpy.reciprocal()
import numpy as np 
a = np.array([0.25, 1.33, 1, 0, 100]) 

print 'Our array is:' 
print a 
print '\n'  

print 'After applying reciprocal function:' 
print np.reciprocal(a) 
print '\n'  

b = np.array([100], dtype = int) 
print 'The second array is:' 
print b 
print '\n'  

print 'After applying reciprocal function:' 
print np.reciprocal(b) 

##numpy.power()
import numpy as np 
a = np.array([10,100,1000]) 

print 'Our array is:' 
print a 
print '\n'  

print 'Applying power function:' 
print np.power(a,2) 
print '\n'  

print 'Second array:' 
b = np.array([1,2,3]) 
print b 
print '\n'  

print 'Applying power function again:' 
print np.power(a,b)

##numpy.mod()
import numpy as np 
a = np.array([10,20,30]) 
b = np.array([3,5,7]) 

print 'First array:' 
print a 
print '\n'  

print 'Second array:' 
print b 
print '\n'  

print 'Applying mod() function:' 
print np.mod(a,b) 
print '\n'  

print 'Applying remainder() function:' 
print np.remainder(a,b) 

import numpy as np 
a = np.array([-5.6j, 0.2j, 11. , 1+1j]) 

print 'Our array is:' 
print a 
print '\n'  

print 'Applying real() function:' 
print np.real(a) 
print '\n'  

print 'Applying imag() function:' 
print np.imag(a) 
print '\n'  

print 'Applying conj() function:' 
print np.conj(a) 
print '\n'  

print 'Applying angle() function:' 
print np.angle(a) 
print '\n'  

print 'Applying angle() function again (result in degrees)' 
print np.angle(a, deg = True)


###Statistical Functions https://www.tutorialspoint.com/numpy/numpy_statistical_functions.htm
##numpy.amin() and numpy.amax()
import numpy as np 
a = np.array([[3,7,5],[8,4,3],[2,4,9]]) 

print 'Our array is:' 
print a  
print '\n'  

print 'Applying amin() function:' 
print np.amin(a,1) 
print '\n'  

print 'Applying amin() function again:' 
print np.amin(a,0) 
print '\n'  

print 'Applying amax() function:' 
print np.amax(a) 
print '\n'  

print 'Applying amax() function again:' 
print np.amax(a, axis = 0)


##numpy.ptp()
import numpy as np 
a = np.array([[3,7,5],[8,4,3],[2,4,9]]) 
print 'Our array is:' 
print a 
print '\n'  
print 'Applying ptp() function:' 
print np.ptp(a) 
print '\n'  
print 'Applying ptp() function along axis 1:' 
print np.ptp(a, axis = 1) 
print '\n'   
print 'Applying ptp() function along axis 0:'
print np.ptp(a, axis = 0) 


##numpy.percentile()
import numpy as np 
a = np.array([[30,40,70],[80,20,10],[50,90,60]]) 
print 'Our array is:' 
print a 
print '\n'  
print 'Applying percentile() function:' 
print np.percentile(a,50) 
print '\n'  
print 'Applying percentile() function along axis 1:' 
print np.percentile(a,50, axis = 1) 
print '\n'  
print 'Applying percentile() function along axis 0:' 
print np.percentile(a,50, axis = 0)


##numpy.median()
import numpy as np 
a = np.array([[30,65,70],[80,95,10],[50,90,60]]) 

print 'Our array is:' 
print a 
print '\n'  

print 'Applying median() function:' 
print np.median(a) 
print '\n'  

print 'Applying median() function along axis 0:' 
print np.median(a, axis = 0) 
print '\n'  
 
print 'Applying median() function along axis 1:' 
print np.median(a, axis = 1)


##numpy.mean()
import numpy as np 
a = np.array([[1,2,3],[3,4,5],[4,5,6]]) 

print 'Our array is:' 
print a 
print '\n'  

print 'Applying mean() function:' 
print np.mean(a) 
print '\n'  

print 'Applying mean() function along axis 0:' 
print np.mean(a, axis = 0) 
print '\n'  

print 'Applying mean() function along axis 1:' 
print np.mean(a, axis = 1)


##numpy.average()
import numpy as np 
a = np.array([1,2,3,4]) 
print 'Our array is:' 
print a 
print '\n'  
print 'Applying average() function:' 
print np.average(a) 
print '\n'  
# this is same as mean when weight is not specified 
wts = np.array([4,3,2,1]) 
print 'Applying average() function again:' 
print np.average(a,weights = wts) 
print '\n'  
# Returns the sum of weights, if the returned parameter is set to True. 
print 'Sum of weights' 
print np.average([1,2,3, 4],weights = [4,3,2,1], returned = True)

import numpy as np 
a = np.arange(6).reshape(3,2) 

print 'Our array is:' 
print a 
print '\n'  

print 'Modified array:' 
wt = np.array([3,5]) 
print np.average(a, axis = 1, weights = wt) 
print '\n'  

print 'Modified array:' 
print np.average(a, axis = 1, weights = wt, returned = True)

##Standard Deviation
std = sqrt(mean(abs(x - x.mean())**2))

##Variance
import numpy as np 
print np.var([1,2,3,4])


###Sort, Search & Counting Functions   
https://www.tutorialspoint.com/numpy/numpy_sort_search_counting_functions.htm
##numpy.sort()
import numpy as np  
a = np.array([[3,7],[9,1]]) 

print 'Our array is:' 
print a 
print '\n'

print 'Applying sort() function:' 
print np.sort(a) 
print '\n' 
  
print 'Sort along axis 0:' 
print np.sort(a, axis = 0) 
print '\n'  

# Order parameter in sort function 
dt = np.dtype([('name', 'S10'),('age', int)]) 
a = np.array([("raju",21),("anil",25),("ravi", 17), ("amar",27)], dtype = dt) 

print 'Our array is:' 
print a 
print '\n'  

print 'Order by name:' 
print np.sort(a, order = 'name')

##numpy.argsort()
import numpy as np 
x = np.array([3, 1, 2]) 

print 'Our array is:' 
print x 
print '\n'  

print 'Applying argsort() to x:' 
y = np.argsort(x) 
print y 
print '\n'  

print 'Reconstruct original array in sorted order:' 
print x[y] 
print '\n'  

print 'Reconstruct the original array using loop:' 
for i in y: 
   print x[i],

##numpy.lexsort()
import numpy as np 

nm = ('raju','anil','ravi','amar') 
dv = ('f.y.', 's.y.', 's.y.', 'f.y.') 
ind = np.lexsort((dv,nm)) 

print 'Applying lexsort() function:' 
print ind 
print '\n'  

print 'Use this index to get sorted data:' 
print [nm[i] + ", " + dv[i] for i in ind] 

##numpy.argmax() and numpy.argmin()
import numpy as np 
a = np.array([[30,40,70],[80,20,10],[50,90,60]]) 

print 'Our array is:' 
print a 
print '\n' 

print 'Applying argmax() function:' 
print np.argmax(a) 
print '\n'  

print 'Index of maximum number in flattened array' 
print a.flatten() 
print '\n'  

print 'Array containing indices of maximum along axis 0:' 
maxindex = np.argmax(a, axis = 0) 
print maxindex 
print '\n'  

print 'Array containing indices of maximum along axis 1:' 
maxindex = np.argmax(a, axis = 1) 
print maxindex 
print '\n'  

print 'Applying argmin() function:' 
minindex = np.argmin(a) 
print minindex 
print '\n'  
   
print 'Flattened array:' 
print a.flatten()[minindex] 
print '\n'  

print 'Flattened array along axis 0:' 
minindex = np.argmin(a, axis = 0) 
print minindex
print '\n'

print 'Flattened array along axis 1:' 
minindex = np.argmin(a, axis = 1) 
print minindex

##numpy.nonzero()
import numpy as np 
a = np.array([[30,40,0],[0,20,10],[50,0,60]]) 

print 'Our array is:' 
print a 
print '\n'  

print 'Applying nonzero() function:' 
print np.nonzero (a)

##numpy.where()
import numpy as np 
x = np.arange(9.).reshape(3, 3) 

print 'Our array is:' 
print x  

print 'Indices of elements > 3' 
y = np.where(x > 3) 
print y  

print 'Use these indices to get elements satisfying the condition' 
print x[y]

##numpy.extract()
import numpy as np 
x = np.arange(9.).reshape(3, 3) 

print 'Our array is:' 
print x  

# define a condition 
condition = np.mod(x,2) == 0 

print 'Element-wise value of condition' 
print condition  

print 'Extract elements using condition' 
print np.extract(condition, x)


###NumPy - Byte Swapping
https://www.tutorialspoint.com/numpy/numpy_byte_swapping.htm
import numpy as np 
a = np.array([1, 256, 8755], dtype = np.int16) 

print 'Our array is:' 
print a  

print 'Representation of data in memory in hexadecimal form:'  
print map(hex,a)  
# byteswap() function swaps in place by passing True parameter 

print 'Applying byteswap() function:' 
print a.byteswap(True) 

print 'In hexadecimal form:' 
print map(hex,a) 
# We can see the bytes being swapped


###NumPy - Copies & Views
https://www.tutorialspoint.com/numpy/numpy_copies_and_views.htm

import numpy as np 
a = np.arange(6) 

print 'Our array is:' 
print a  

print 'Applying id() function:' 
print id(a)  

print 'a is assigned to b:' 
b = a 
print b  

print 'b has same id():' 
print id(b)  
print 'Change shape of b:' 
b.shape = 3,2 
print b  
print 'Shape of a also gets changed:' 
print a

##View or Shallow Copy
import numpy as np 
# To begin with, a is 3X2 array 
a = np.arange(6).reshape(3,2) 

print 'Array a:' 
print a  

print 'Create view of a:' 
b = a.view() 
print b  

print 'id() for both the arrays are different:' 
print 'id() of a:'
print id(a)  
print 'id() of b:' 
print id(b)  

# Change the shape of b. It does not change the shape of a 
b.shape = 2,3 

print 'Shape of b:' 
print b  
print 'Shape of a:' 
print a

import numpy as np 
a = np.array([[10,10], [2,3], [4,5]]) 
print 'Our array is:' 
print a  
print 'Create a slice:' 
s = a[:, :2] 
print s 

##Deep Copy
import numpy as np 
a = np.array([[10,10], [2,3], [4,5]]) 

print 'Array a is:' 
print a  

print 'Create a deep copy of a:' 
b = a.copy() 
print 'Array b is:' 
print b 

#b does not share any memory of a 
print 'Can we write b is a' 
print b is a  

print 'Change the contents of b:' 
b[0,0] = 100 

print 'Modified array b:' 
print b  

print 'a remains unchanged:' 
print a


###NumPy - Matrix Library
NumPy package contains a Matrix library numpy.matlib. This module has functions that return matrices instead of ndarray objects.
https://www.tutorialspoint.com/numpy/numpy_matrix_library.htm

##matlib.empty()
The matlib.empty() function returns a new matrix without initializing the entries. The function takes the following parameters
numpy.matlib.empty(shape, dtype, order)

import numpy.matlib 
import numpy as np 
print np.matlib.empty((2,2)) 
# filled with random data

import numpy.matlib 
import numpy as np 
print np.matlib.zeros((2,2)) 

import numpy.matlib 
import numpy as np 
print np.matlib.ones((2,2))

##numpy.matlib.eye()
This function returns a matrix with 1 along the diagonal elements and the zeros elsewhere. The function takes the following parameters.
numpy.matlib.eye(n, M,k, dtype)

import numpy.matlib 
import numpy as np 
print np.matlib.eye(n = 3, M = 4, k = 0, dtype = float)

import numpy.matlib 
import numpy as np 
print np.matlib.identity(5, dtype = float)

import numpy.matlib 
import numpy as np 
print np.matlib.rand(3,3)

import numpy.matlib 
import numpy as np  
i = np.matrix('1,2;3,4') 
print i 

import numpy.matlib 
import numpy as np  
j = np.asarray(i) 
print j 

import numpy.matlib 
import numpy as np  
k = np.asmatrix (j) 
print k


###NumPy - Linear Algebra
https://www.tutorialspoint.com/numpy/numpy_linear_algebra.htm
NumPy package contains numpy.linalg module that provides all the functionality required for linear algebra. 
Some of the important functions in this module are described in the following table.

dot, vdot, inner, matmul, determinant, solve, inv


###NumPy - Matplotlib
https://www.tutorialspoint.com/numpy/numpy_matplotlib.htm

import numpy as np 
from matplotlib import pyplot as plt 
x = np.arange(1,11) 
y = 2 * x + 5 
plt.title("Matplotlib demo") 
plt.xlabel("x axis caption") 
plt.ylabel("y axis caption") 
plt.plot(x,y) 
plt.show()

import numpy as np 
from matplotlib import pyplot as plt 
x = np.arange(1,11) 
y = 2 * x + 5 
plt.title("Matplotlib demo") 
plt.xlabel("x axis caption") 
plt.ylabel("y axis caption") 
plt.plot(x,y,"ob") 
plt.show() 

##Sine Wave Plot
import numpy as np 
import matplotlib.pyplot as plt  

# Compute the x and y coordinates for points on a sine curve 
x = np.arange(0, 3 * np.pi, 0.1) 
y = np.sin(x) 
plt.title("sine wave form") 

# Plot the points using matplotlib 
plt.plot(x, y) 
plt.show() 

##subplot()
import numpy as np 
import matplotlib.pyplot as plt  
   
# Compute the x and y coordinates for points on sine and cosine curves 
x = np.arange(0, 3 * np.pi, 0.1) 
y_sin = np.sin(x) 
y_cos = np.cos(x)  
   
# Set up a subplot grid that has height 2 and width 1, 
# and set the first such subplot as active. 
plt.subplot(2, 1, 1)
   
# Make the first plot 
plt.plot(x, y_sin) 
plt.title('Sine')  
   
# Set the second subplot as active, and make the second plot. 
plt.subplot(2, 1, 2) 
plt.plot(x, y_cos) 
plt.title('Cosine')  
   
# Show the figure. 
plt.show()

##bar()
from matplotlib import pyplot as plt 
x = [5,8,10] 
y = [12,16,6]  

x2 = [6,9,11] 
y2 = [6,15,7] 
plt.bar(x, y, align = 'center') 
plt.bar(x2, y2, color = 'g', align = 'center') 
plt.title('Bar graph') 
plt.ylabel('Y axis') 
plt.xlabel('X axis')  
plt.show()


###NumPy - Histogram Using Matplotlib
https://www.tutorialspoint.com/numpy/numpy_histogram_using_matplotlib.htm

##numpy.histogram()
import numpy as np 
a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27]) 
np.histogram(a,bins = [0,20,40,60,80,100]) 
hist,bins = np.histogram(a,bins = [0,20,40,60,80,100]) 
print hist 
print bins 

##plt()
from matplotlib import pyplot as plt 
import numpy as np  
a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27]) 
plt.hist(a, bins = [0,20,40,60,80,100]) 
plt.title("histogram") 
plt.show()


###I/O with NumPy
https://www.tutorialspoint.com/numpy/numpy_with_io.htm

##numpy.save()
import numpy as np 
a = np.array([1,2,3,4,5]) 
np.save('outfile',a)

import numpy as np 
b = np.load('outfile.npy') 
print b 

##savetxt()
import numpy as np 

a = np.array([1,2,3,4,5]) 
np.savetxt('out.txt',a) 
b = np.loadtxt('out.txt') 
print b 

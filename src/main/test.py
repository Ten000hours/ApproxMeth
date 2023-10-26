import torch

from ctypes import c_float, c_int32, cast, byref, POINTER

def ctypes_isqrt(number):
    threehalfs = 1.5
    x2 = number * 0.5
    for elem in number:
        y = c_float(elem)

        i = cast(byref(y), POINTER(c_int32)).contents.value
        i = c_int32(0x5f3759df - (i >> 1))
        y = cast(byref(i), POINTER(c_float)).contents.value

        y = y * (1.5 - (0.5*elem * y * y))
        number = torch.where(number==elem, y, number)
    return number
# Define a custom function
def custom_function(x):
    # Example: Square the input
    return x * x

# Create a sample tensor
tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])

# Apply the custom function to each element in the tensor
result = ctypes_isqrt(tensor)

# Print the result
print(result)

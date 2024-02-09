import random
import time
import matplotlib.pyplot as plt
import numpy as nump
from tabulate import tabulate

def insertion_sort(b):
    for i in range(1, len(b)):
        up = b[i]
        j = i - 1
        while j >= 0 and b[j] > up:
            b[j + 1] = b[j]
            j -= 1
        b[j + 1] = up
    return b

def quick_sort(arr,k):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left,k) + middle + quick_sort(right,k)

def merge_sort(arr,k):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid],k)
    right = merge_sort(arr[mid:],k)
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def heap_sort(arr,k):
    def heapify(arr, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and arr[left] > arr[largest]:
            largest = left
        if right < n and arr[right] > arr[largest]:
            largest = right

        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)

    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
    return arr

# Using counting sort to sort the elements in the basis of significant places
def counting_sort(array, place):
    size = len(array)
    output = [0] * size
    count = [0] * 10

    # Calculate count of elements
    for i in range(0, size):
        index = array[i] // place
        count[index % 10] += 1

    # Calculate cumulative count
    for i in range(1, 10):
        count[i] += count[i - 1]

    # Place the elements in sorted order
    i = size - 1
    while i >= 0:
        index = array[i] // place
        output[count[index % 10] - 1] = array[i]
        count[index % 10] -= 1
        i -= 1

    for i in range(0, size):
        array[i] = output[i]


# Main function to implement radix sort
def radix_sort(array, size):
    # Get maximum element
    max_element = max(array)

    # Apply counting sort to sort elements based on place value.
    place = 1
    while max_element // place > 0:
        counting_sort(array, place)
        place *= 10

    return array


def counting_sort(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    for i in range(n):
        index = arr[i] // exp
        count[index % 10] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    i = n - 1
    while i >= 0:
        index = arr[i] // exp
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1

    for i in range(n):
        arr[i] = output[i]
def insertion_sort_tim(arr, left=0, right=None):
    if right is None:
        right = len(arr) - 1

    for i in range(left + 1, right + 1):
        key = arr[i]
        j = i - 1
        while j >= left and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def merge_tim(arr, left, mid, right):
    len_left = mid - left + 1
    len_right = right - mid

    left_part = arr[left:left + len_left]
    right_part = arr[mid + 1:mid + 1 + len_right]

    i = j = 0
    k = left

    while i < len_left and j < len_right:
        if left_part[i] <= right_part[j]:
            arr[k] = left_part[i]
            i += 1
        else:
            arr[k] = right_part[j]
            j += 1
        k += 1

    while i < len_left:
        arr[k] = left_part[i]
        i += 1
        k += 1

    while j < len_right:
        arr[k] = right_part[j]
        j += 1
        k += 1


def tim_sort(arr,k):
    min_run = 32
    n = len(arr)

    # Sort individual subarrays of size min_run
    for i in range(0, n, min_run):
        insertion_sort_tim(arr, i, min((i + min_run - 1), n - 1))

    # Merge subarrays in chunks
    size = min_run
    while size < n:
        for left in range(0, n, 2 * size):
            mid = min(n - 1, left + size - 1)
            right = min((left + 2 * size - 1), (n - 1))
            merge_tim(arr, left, mid, right)
        size *= 2  # Double the size for the next iteration

    return arr

def bucket_sort(arr,k):
    # Find the maximum and minimum values in the array
    max_val = max(arr)
    min_val = min(arr)

    # Calculate the range of values in the array
    arr_range = max_val - min_val + 1

    # Define the number of buckets based on the range of values
    num_buckets = min(arr_range, len(arr))  # Number of buckets is at most the length of the array

    # Calculate the range of each bucket
    bucket_range = (arr_range + num_buckets - 1) // num_buckets  # Use integer division

    # Create empty buckets
    buckets = [[] for _ in range(num_buckets)]

    # Assign each element to a bucket
    for num in arr:
        index = min((num - min_val) // bucket_range, num_buckets - 1)  # Ensure index doesn't exceed num_buckets - 1
        buckets[index].append(num)

    # Sort each bucket individually (using another sorting algorithm like insertion sort)
    for bucket in buckets:
        insertion_sort(bucket)

    # Concatenate all buckets to get the sorted array
    sorted_arr = [num for bucket in buckets for num in bucket]
    
    return sorted_arr

def generate_data(size,k, choice):
  if (choice == 1):
     return [random.randint(0, size) for _ in range(size)]
  elif(choice ==2):
     return [random.randint(0, k) for _ in range(size)]
  elif (choice ==3):
     return [random.randint(0, size**3) for _ in range(size)]
  elif (choice==4):
     return [random.randint(0, int(nump.log2(size))) for _ in range(size)]
  elif (choice == 5):
    if (size > 10000):
      return [random.randint(0, size/1000) * 1000 for _ in range(size)]
    else:
      return [random.randint(0, size) * 1000 for _ in range(size)]
  elif (choice ==6):
    data = list(range(size))
    for _ in range(int(nump.log2(size)/2)):
      i, j = random.sample(range(size), 2)
      data[i], data[j] = data[j], data[i]
    return data

def measure_time(sort_function, data, size, choice):
    start_time = time.time()
    if(choice == 3):
      sorted_data = sort_function(data,size*1000)
    else:
      sorted_data = sort_function(data,size)
    end_time = time.time()
    return end_time - start_time

def plot_graph(sizes, times, algorithm_names):
    for i in range(len(algorithm_names)):
        plt.plot(sizes, times[i], label=algorithm_names[i])
    plt.xlabel('Input Size')
    plt.ylabel('Time (seconds)')
    plt.title('Sorting Algorithm Performance')
    plt.legend()
    plt.show()

def take_user_input():
    k = choice = 0
    print("Select the distribution to be analyzed:")
    print("1. n randomly chosen integers in the range [0 ... n]")
    print("2. n randomly chosen integers in the range [0 ... k], k < 1000")
    print("3. n randomly chosen integers in the range [0 ... n^3]")
    print("4. n randomly chosen integers in the range [0 ... log n]")
    print("5. n randomly chosen integers that are multiples of 1000 in the range [0 ... n]")
    print("6. the in order integers [0 ... n] where (log n)/2 randomly chosen values have been swapped with another value")
    choice = int(input("Enter your choice: "))
    if (choice not in range(1,7) ):
      print ("Wrong choice!")
      exit()
    else:
      if(choice == 2):
        k = int(input("Enter the value of k: "))
    return k, choice

# Example usage
input_sizes = [100000,200000,500000,800000,1000000]
k, choice= take_user_input()
algorithms = [quick_sort, merge_sort, heap_sort, radix_sort, tim_sort,bucket_sort]
Algo_legends = ['Quick_Sort', 'Merge_Sort', 'Heap_Sort', 'Radix_Sort', 'Tim_Sort','Bucket_Sort']

times = [[] for _ in range(len(algorithms))]


for input_size in input_sizes:
  data = generate_data(input_size,k,choice)
  for i, algorithm in enumerate(algorithms):
    elapsed_time = measure_time(algorithm, data,input_size,choice)
    times[i].append(elapsed_time)

plot_graph( input_sizes, times, Algo_legends)
print ("Algorithm",end = " ")
for i in range (0, len(input_sizes)):
  print ("Size",end="_")
  print (input_sizes[i],end = " ")
print ("\n")
for i in range(0, len(Algo_legends)):
  print(Algo_legends[i],end = " ")
  for j in range (0, len(input_sizes)):
    print(times[i][j], end = " ")
  print("\n")

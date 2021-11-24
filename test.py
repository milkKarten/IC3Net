import os, sys
numbers = [2,1,3,4,7]
more_numbers = [*numbers, 11, 18]
print(*more_numbers, sep=', ')

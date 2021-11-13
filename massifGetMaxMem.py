#!/usr/bin/python3
import sys
import re
import os

heapArr = []
stackArr = []
with open(sys.argv[1], "r") as f: 
    for line in f:
        heap = params = re.search('mem_heap_B=[0-9]+$',line)
        if heap is not None:
            heap = re.sub('mem_heap_B=', '', heap.group(0))
            heapArr.append(int(heap))
        stack = params = re.search('mem_stacks_B=[0-9]+$',line)
        if stack is not None:
            stack = re.sub('mem_stacks_B=', '', stack.group(0))
            stackArr.append(int(stack))
maxHeap = max(heapArr)
maxStack = max(stackArr)
heapB410 = heapArr[-10]
stackB410 = stackArr[-10]
print ("%0.3f, %0.3f"%(maxHeap/1024/1024, heapB410/1024/1024), end='')
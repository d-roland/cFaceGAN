import os
import numpy as np

from csv import reader

celeba_dir = 'C:/celeba-dataset'

with open(os.path.join(celeba_dir, 'list_attr_celeba.csv'), 'r') as read_obj:
    csv_reader = reader(read_obj)
    header = next(csv_reader)
#    index = header.index(labels_to_extract[0]) # for now
    width = len(header)-1
    counts = np.zeros(width)
    for row in csv_reader:
        for column in range(width):
            counts[column] += row[column+1] == '1'

counts /= 202599.
for index in range(width):
    print(str(index)+":"+header[index+1]+":"+str(counts[index]))
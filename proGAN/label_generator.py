import os
import numpy as np

from csv import reader

celeba_dir = 'C:/celeba-dataset'
# labels_to_extract = ['Male', "Smiling", "Attractive", "Heavy_Makeup"]
labels_to_extract = ['All']

for label in labels_to_extract:
    labels = np.zeros(202599, dtype=np.uint8)
    with open(os.path.join(celeba_dir, 'list_attr_celeba.csv'), 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        if(label == "All"):
            i = 0
            labels = np.zeros((202599, len(header)-1), dtype=np.uint8)
            for row in csv_reader:
                for input_label in header[1:]:
                    index = header.index(input_label)  # for now
                    labels[i,index-1] = row[index] == '1'
                i += 1
            for label_index in range(labels.shape[1]):
                labels[:,label_index].tofile(os.path.join(celeba_dir, 'attr' + str(label_index)))
        else:
            index = header.index(label) # for now
            i = 0
            for row in csv_reader:
                labels[i] = row[index] == '1'
                i += 1
            labels.tofile(os.path.join(celeba_dir, 'attr'+str(index)))

# labels2 = np.fromfile(os.path.join(celeba_dir, 'attr'+str(index)), dtype=np.uint8)
#
# assert labels2.shape == (202599,) and labels2.dtype == np.uint8
# onehot = np.zeros((labels2.size, np.max(labels2) + 1), dtype=np.float32)
# onehot[np.arange(labels2.size), labels2] = 1.0

import os
import imageio
import numpy as np

class CelebA:
    def getData(self, DATA_DIR, num_images):
        list = os.listdir(DATA_DIR)
        list.sort()
        images = []
        for image_index in range(num_images):
            f = list[image_index]
            image = imageio.imread(os.path.join(DATA_DIR, f))
            images.append(image)
            if(image_index % 1000 == 0):
                print("loaded "+str(image_index))
        return np.array(images)


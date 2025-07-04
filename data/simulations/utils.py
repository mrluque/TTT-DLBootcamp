import numpy as np
from matplotlib import pyplot as plt

def show_images(images, titles = [], nRow = 1, nCol = 1):
    '''
    Show multiple images using matplotlib
    
    @param images: List with multiple images which will be showed in the figure
    @param titles: List with the titles of each image
    @param nRow: number of rows of the figure
    @param nCol: number of columns of the figure
    '''
    plt.figure()
    for i in range(0, len(images)):
        plt.subplot(nRow,nCol,i+1)
        plt.imshow(images[i], "gray", interpolation = 'bicubic')
        plt.xticks([]),plt.yticks([])
        if titles:
            plt.title(titles[i])
        
    plt.show()

def masks_to_colorimg(masks):
    '''
    Generate a color image from masks array of size (nMask, heigh, width)
    '''
    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    channels, height, width = masks.shape

    colors = (np.random.rand(channels, 3) * 255).astype(np.uint8)
    
    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:,y,x] > 0.5]

            if len(selected_colors) > 0:
                colorimg[y,x,:] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8)

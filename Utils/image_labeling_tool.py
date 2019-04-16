import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

SOURCE_DIR = './' #The path to Non-Labeled images
LABELED_DIR = './labeled' #Where to save the labeled images
DELETE = True #If true when 9 is pressed the image is removed otherwise it's left in the source directory

print(" - Image Classification Tool - ")
print("Source Directory: " + SOURCE_DIR)
print("Labeled Images Directory: " + LABELED_DIR)
if(DELETE):
    print("Delete Behaviour: Delete File")
else:
    print("Delete Behaviour: Keep File")
print("Keys: 0 (Negative), 1 (Neutral), 2 (Positive), 9(Delete)\n")

fig,ax = plt.subplots(1,1)
images_filenames = os.listdir(os.path.join(SOURCE_DIR, 'images'))
total_images = len(images_filenames)

for i, filename in enumerate(images_filenames):
    if(filename != '.DS_Store'):
        filename_absolute = os.path.join(SOURCE_DIR, 'images', filename)

        #Show the image
        image = mpimg.imread(filename_absolute)
        im = ax.imshow(image)
        im.set_data(image)
        fig.canvas.draw_idle()
        plt.show(block=False)

        #Read input
        action = input('Image {}/{}: '.format(str(i+1), str(total_images)))
        while action not in ['0', '1', '2', '9']:
            print("Chose a valid characther! 0 (Negative), 1 (Neutral), 2 (Positive), 9(Delete)")
            action = input("Image " + str(i) + ": ")
        
        #Rename or Delete
        if(action=='9'):
            if(DELETE):
                os.remove(filename_absolute)
        else:
            new_filename = action + '_' + filename
            os.rename(filename_absolute, os.path.join(LABELED_DIR, new_filename))

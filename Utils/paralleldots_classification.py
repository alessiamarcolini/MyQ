import os
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

DATASET_DIR = '../../myq_dataset/Selfie-dataset/images/'

if os.path.exists(os.path.join(DATASET_DIR, '.DS_Store')):
    os.remove(os.path.join(DATASET_DIR, '.DS_Store'))

images_filenames = os.listdir(DATASET_DIR)

for i, filename in enumerate(images_filenames):
    filename_absolute = os.path.join(DATASET_DIR, filename)
    #print(filename_absolute)
    multipart_data = MultipartEncoder(fields={ 'file': (filename, open(filename_absolute, 'rb'), 'image/jpg')})
    
    response = requests.post('https://www.paralleldots.com/visual/facial/emotion', data=multipart_data, headers={'Content-Type': multipart_data.content_type})
    print(response.text)
    print(filename)
    if i == 20:
        break

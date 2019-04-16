import os
import ast
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

DATASET_DIR = '../../myq_dataset/Selfie-dataset/'
AUTOMATIC_LABEL_DIR = os.path.join(DATASET_DIR, 'automatic_labelling')
MANUAL_LABEL_DIR = os.path.join(DATASET_DIR, 'manual_labelling')

if os.path.exists(os.path.join(DATASET_DIR, 'images' '.DS_Store')):
    os.remove(os.path.join(DATASET_DIR, 'images', '.DS_Store'))

images_filenames = os.listdir(os.path.join(DATASET_DIR, 'images'))

emotions_mapping = {'Sad': '0', 'Fear': '0' , 'Disgust': '0', 'Angry': '0', 'Neutral': '1', 'Happy': '2', 'Surprise': '2' }

os.makedirs(os.path.join(DATASET_DIR, MANUAL_LABEL_DIR), exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, AUTOMATIC_LABEL_DIR), exist_ok=True)
    

for i, filename in enumerate(images_filenames):
    filename_absolute = os.path.join(DATASET_DIR, 'images', filename)
    #print(filename_absolute)
    multipart_data = MultipartEncoder(fields={ 'file': (filename, open(filename_absolute, 'rb'), 'image/jpg')})
    
    response = requests.post('https://www.paralleldots.com/visual/facial/emotion', data=multipart_data, headers={'Content-Type': multipart_data.content_type})
    print(response.text)
    #print(filename)
    
    d = ast.literal_eval(response.text)
    if 'output' in d: # no face detected
        print('no output')
        os.rename(filename_absolute, os.path.join(MANUAL_LABEL_DIR, filename))

    else:
        emotions = d['facial_emotion']
        em_higher_score = ''
        higher_score = -1
        for entry in emotions:
            if entry['score'] > higher_score:
                higher_score = entry['score']
                em_higher_score = entry['tag']
        print(em_higher_score)
        new_filename = emotions_mapping[em_higher_score] + '_' + filename
        os.path.join(DATASET_DIR, 'images', new_filename)
        print(new_filename)
        os.rename(filename_absolute, os.path.join(AUTOMATIC_LABEL_DIR, new_filename))
        
            
    

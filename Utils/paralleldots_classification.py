import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

multipart_data = MultipartEncoder(
  fields={ 'file': ('test4.jpeg', open('test4.jpeg', 'rb'), 'image/jpeg')}
)

response = requests.post( 'https://www.paralleldots.com/visual/facial/emotion',
                         data=multipart_data,
                         headers={'Content-Type': multipart_data.content_type})

print(response.text)

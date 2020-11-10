import requests
import io
import cv2
def plate_reader(img):
    _, buffer = cv2.imencode(".jpg", img)
    io_buf = io.BytesIO(buffer)
    token="1d3d7517ff820f00f7a4c9d0404aef2eacce850f"
    token2="2dd4ee15c28eafec3e463d37885cd9d644ae7f11"
    regions = ['in', 'it'] # Change to your country
    response = requests.post(
        'https://api.platerecognizer.com/v1/plate-reader/',
        data=dict(regions=regions),  # Optional
        files=dict(upload=io_buf),
        headers={'Authorization': 'Token 2dd4ee15c28eafec3e463d37885cd9d644ae7f11 '})
    out=response.json()
    try:
        if out['results']==[]:
             return "Can't read"
        num_plate = out['results'][0]['plate']
        score=out['results'][0]['score']
    except:
        print("Exception encountered")
        num_plate=" "
        score=0
    return num_plate,score

# img = cv2.imread('D:\\Programs\\Final Project\\code_output\\ka03hv9475\\num_plate.jpg')
# print(plate_reader(img))


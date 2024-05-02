from transformers import AutoModel
from PIL import Image
import pdb

image_path = '/afs/csail.mit.edu/u/s/sophiejg/public/public/Rology-dataset/Images/657ac2a5880dc800080d649d_1.jpg'

model = AutoModel.from_pretrained("ECOFRI/CXR-LLAVA-v2", trust_remote_code=True)
print('Loaded Model')
model = model.to("cuda")
pdb.set_trace()
cxr_image = Image.open(image_path)
response = model.write_radiologic_report(cxr_image)
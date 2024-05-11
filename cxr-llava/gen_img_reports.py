from transformers import AutoModel
from PIL import Image
import pdb
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import torch
import pandas as pd
import zipfile
import time

print('hi')

print(torch.cuda.is_available())
PATH = '/om/user/sophiejg/project/mlhc_rology/rology_CXR_LLaVA/'
model = AutoModel.from_pretrained("ECOFRI/CXR-LLAVA-v2", trust_remote_code=True)
print('Finished downloading')

model = model.to("cuda")
print(model)


xray_df = pd.read_csv(PATH+'dataset/rology_batch1.csv')
xray_df['generated_report'] = pd.Series(dtype=str)
xray_df.head()



def custom_radiologic_report(model, image, temperature=0.2, top_p=0.8):
        # chat = [
        #     {"role": "system",
        #      "content": "You are a helpful radiologist. Try to interpret chest x ray image and answer to the question that user provides."},
        #     {"role": "user",
        #      "content": "<image>\nWrite a radiologic report on the given chest radiograph. Provide the output in a format similar to the given example:\
        #                 Lungs: Normal both lung volumes with fair aeration. No focal lesions. Normal tracheobronchial tree.\
        #                 Mediastinum: Central. No hilar lymphadenopathy.\
        #                 Heart: Normal size and configuration. \
        #                 Pleural spaces: Clear both costophrenic angles.\
        #                 The Diaphragm: Normal position and configuration of both diaphragmatic copulae.\
        #                 Osseous structures: Normal bony thorax.\
        #                 Include information about atelectasis,cardiomegaly, consolidation, pulmonary edema, pleural effusion, and pneumothorax.\n"}
        # ]

        # chat = [
        #     {"role": "system",
        #      "content": "You are a helpful radiologist. Try to interpret chest x ray image and answer to the question that user provides."},
        #     {"role": "user",
        #      "content": "<image>\nWrite a radiologic report on the given chest radiograph. Provide observations on the lungs, mediastinum,\
        #         heart, pleural spaces, the diaphragm, and osseous structures.\
        #         Include information about atelectasis,cardiomegaly, consolidation, pulmonary edema, pleural effusion, and pneumothorax.\n"}
        # ]

        chat = [
            {"role": "system",
             "content": "You are a helpful radiologist. Try to interpret chest x ray image and answer to the question that user provides."},
            {"role": "user",
             "content": "<image>\nWrite observations on the given chest radiograph.\
                For the lungs, include information on lung volume, focal lesions, and the tracheobronchial tree.\
                For the mediastinum, include information on it's location and hilar lymphadenopathy.\
                For the heart, include information on size and configurations.\
                For the pleural spaces, include information on costophrenic angles.\
                For the diaghram, include information on the position and configurarions.\
                For the osseous structures, include information on the bony thorax."}
        ]
        response = model.generate_cxr_repsonse(chat=chat,image=image, temperature=temperature, top_p=top_p)
        return response



# proces images directly from folder
def process_images(folder_path, df, custom_prompt = False, file_limit=None):
    processed_count = 0  # Track the number of processed files
    # pdb.set_trace()
    for filename in tqdm(os.listdir(folder_path), desc='Processing images', unit='image'):
        image_id = filename.split('_')[0]

        if image_id in df['Study_id'].values:
            image_path = os.path.join(folder_path, filename)
            curr_img = Image.open(image_path)


            # start_gen_time = time.time()
            if custom_prompt:
                 generated_report = custom_radiologic_report(model, curr_img)
            else:
                generated_report = model.write_radiologic_report(curr_img)
            # end_gen_time = time.time()
            df.loc[df['Study_id'] == image_id, 'generated_report'] = generated_report
            # end_df_update = time.time()
            processed_count += 1

            # elapsed_gen_time = end_gen_time - start_gen_time
            # print("Time taken to generate report:", elapsed_gen_time, "seconds")

            # elapsed_df_time = end_df_update - end_gen_time
            # print("Time taken to update df:", elapsed_df_time, "seconds")

            if file_limit is not None and processed_count >= file_limit:
                break  # Exit the loop if the file limit is reached




img_folder_path = 'dataset/Images'
file_limit = None  # Process only the first 5 files for testing
process_images(PATH + img_folder_path, xray_df, custom_prompt= True, file_limit = file_limit)

output_folder_path = 'logs/custom_generated_reports.csv'

xray_df.to_csv(PATH+output_folder_path)









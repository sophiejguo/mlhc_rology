import io
import os
import pandas as pd
import requests
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

import pdb

PATH = '/om/user/sophiejg/project/mlhc_rology/'


# step 1: Setup constant
device = "cuda"
dtype = torch.float16


xray_df = pd.read_csv(PATH+'rology_CXR_LLaVA/dataset/rology_batch1.csv')
xray_df['generated_report'] = pd.Series(dtype=str)
xray_df.head()

# step 2: Load Processor and Model
processor = AutoProcessor.from_pretrained("StanfordAIMI/CheXagent-8b", trust_remote_code=True)
generation_config = GenerationConfig.from_pretrained("StanfordAIMI/CheXagent-8b")
model = AutoModelForCausalLM.from_pretrained("StanfordAIMI/CheXagent-8b", torch_dtype=dtype, trust_remote_code=True).to(device=device)


def custom_radiologic_report(model, image, temperature=0.2, top_p=0.8):
        # chat = [
        #     {"role": "system",
        #      "content": "You are a helpful radiologist. Try to interpret chest x ray image and answer to the question that user provides."},
        #     {"role": "user",
        #      "content": "<image>\nWrite observations on the given chest radiograph.\
        #         For the lungs, include information on lung volume, focal lesions, and the tracheobronchial tree.\
        #         For the mediastinum, include information on it's location and hilar lymphadenopathy.\
        #         For the heart, include information on size and configuration.\
        #         For the pleural spaces, include information on costophrenic angles.\
        #         For the diaghram, include information on the position and configurariont.\
        #         For the osseous structures, include information on the bony thorax."}
        # ]
        prompt = "You are a helpful radiologist. Try to interpret chest x ray image and answer to the question that user provides.\
                Write observations on the given chest radiograph.\
                For the lungs, include information on lung volume, focal lesions, and the tracheobronchial tree.\
                For the mediastinum, include information on it's location and hilar lymphadenopathy.\
                For the heart, include information on size and configuration.\
                For the pleural spaces, include information on costophrenic angles.\
                For the diaghram, include information on the position and configurarion.\
                For the osseous structures, include information on the bony thorax."

        chat = f"USER: <s>{prompt} ASSISTANT: <s>"
        inputs = processor(images=[image], text=f" USER: <s>{prompt} ASSISTANT: <s>", return_tensors="pt").to(device=device, dtype=dtype)
        output = model.generate(**inputs, generation_config=generation_config)[0]
        response = processor.tokenizer.decode(output, skip_special_tokens=True)
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
    
            generated_report = custom_radiologic_report(model, curr_img)
        
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




img_folder_path = 'rology_CXR_LLaVA/dataset/Images'
file_limit = None  # Process only the first 5 files for testing
process_images(PATH + img_folder_path, xray_df, custom_prompt= True, file_limit = file_limit)

output_folder_path = PATH + 'rology_cheXagent/logs/cheXagent_custom_generated_reports.csv'
xray_df.to_csv(output_folder_path)





# step 3: Fetch the images
# image_path = "https://upload.wikimedia.org/wikipedia/commons/3/3b/Pleural_effusion-Metastatic_breast_carcinoma_Case_166_%285477628658%29.jpg"
# images = [Image.open(io.BytesIO(requests.get(image_path).content)).convert("RGB")]

# # step 4: Generate the Findings section
# prompt = f'Describe "Airway"'
# inputs = processor(images=images, text=f" USER: <s>{prompt} ASSISTANT: <s>", return_tensors="pt").to(device=device, dtype=dtype)
# output = model.generate(**inputs, generation_config=generation_config)[0]
# response = processor.tokenizer.decode(output, skip_special_tokens=True)
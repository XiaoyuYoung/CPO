from tqdm import tqdm
import os

## 获取csv文件
def get_csv(file, delimiter=','):
    import csv
    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        result = list(reader)
    return result

def get_json(file):
    import json
    with open(file, 'r', encoding='utf-8') as f:
        dicts = json.load(f)
    return dicts


## save dict as json
def save_json(dicts, file, indent=2):
    import json
    info = json.dumps(dicts, indent=indent, ensure_ascii=False)
    with open(file, 'w', encoding='utf-8') as f:  # 使用.dumps()方法时，要写入
        f.write(info)



items = get_json('/home/xiaoyyan/Data/code/2505-reinforced_drift/data/MIMIC-CXR/counterfact/mimic-counterfact.json')
jsons = []

for idx, item in tqdm(enumerate(items), total = len(items)):

    dicom_id, study_id, subject_id = item['dicom_id'],item['sid'],item['pid'] 

    ## find image
    src_img_dire = '/MIMIC-CXR/data/DR/public/MIMIC'
    for i in range(7):
        if os.path.exists(os.path.join(src_img_dire, 'mimic-cxr-00{}'.format(i), 'images', "{}.jpg".format(dicom_id))):
            img_path = os.path.join(src_img_dire, 'mimic-cxr-00{}'.format(i), 'images', "{}.jpg".format(dicom_id))
            break

    ## find report
    content = item['report']
    disease = item['diagnose']

    for cdis, cfact in item['counterfact'].items():

        dialogue = {
            "messages": [
                {"role": "user", "content": "This is a patient's chest DR image <image>. Please help me determine the patient's disease."},
                {"role": "assistant", "content": "<think> " + content + " </think>" + disease},
            ],
            "images": [img_path],

            "rejected_messages": [
                {"role": "user", "content": "This is a patient's chest DR image <image>. Please help me determine the patient's disease."},
                {"role": "assistant", "content": "<think> " + cfact + " </think> The disease of this patient is " + cdis +"."},
            ],
            "rejected_images": [img_path]     

        }

        jsons.append(dialogue)

        break



save_json(jsons, '/home/xiaoyyan/Data/code/2512-drift/data/mimic-rft-counterfact-reason-cot.json')


    
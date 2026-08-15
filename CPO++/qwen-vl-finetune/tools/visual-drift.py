from tqdm import tqdm
import os
import numpy as np
from pack_data import get_counterfact_img

## 获取csv文件
def get_csv(file, delimiter=','):
    import csv
    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        result = list(reader)
    return result


## save dict as json
def save_json(dicts, file, indent=2):
    import json
    info = json.dumps(dicts, indent=indent, ensure_ascii=False)
    with open(file, 'w', encoding='utf-8') as f:  # 使用.dumps()方法时，要写入
        f.write(info)


def extract_findings(text):
    lines = text.split('\n')
    findings = []
    capture = False
    
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.upper().startswith("FINDINGS:"):
            capture = True
            content_after_label = line.split("FINDINGS:", 1)[-1].strip()
            if content_after_label:
                findings.append(content_after_label)
            continue
        elif stripped_line.upper().startswith("IMPRESSION:"):
            capture = False
        elif capture:
            findings.append(line)
    
    return ''.join(findings).strip()





items = get_csv('/home/xiaoyyan/Data/data/MIMIC-CXR/mimic-counterfact-visual.json')
dis_list = items[0][4:]


jsons = []
items = items[1:]


for idx, item in tqdm(enumerate(items), total = len(items)):

    dicom_id, study_id, subject_id = item[1:4]

    ## find image
    src_img_dire = '/home/xiaoyyan/Data/xiaoyu/data/MIMIC-CXR/data/DR/public/MIMIC'
    for i in range(7):
        if os.path.exists(os.path.join(src_img_dire, 'mimic-cxr-00{}'.format(i), 'images', "{}.jpg".format(dicom_id))):
            img_path = os.path.join(src_img_dire, 'mimic-cxr-00{}'.format(i), 'images', "{}.jpg".format(dicom_id))
            break

    ## find counterfact img
    cf_img_dicom_id = get_counterfact_img(idx, items)   
    cf_img_path = os.path.join(src_img_dire, 'mimic-cxr-00{}'.format(i), 'images', "{}.jpg".format(cf_img_dicom_id))    
    
    ## find report
    src_txt_dire = '/home/xiaoyyan/Data/xiaoyu/data/MIMIC-CXR/reports'
    txt_path = os.path.join(src_txt_dire, 'p{}'.format(study_id[:2]), 'p{}'.format(study_id), 's{}.txt'.format(subject_id))
    with open(txt_path, "r", encoding="utf-8") as file:
        content = file.read()

    content = extract_findings(content)
    # print("{} {}".format(idx, content))

    if content == '':
        continue


    ## find disease
    dis_idx = list(map(float, item[4:]))
    disease = " The disease of this patient is " + ", ".join(dis for dis, id in zip(dis_list, dis_idx) if id > 0.5) + '.'

    dialogue = {
        "messages": [
            {"role": "user", "content": "This is a patient's chest DR image <image>. Please help me determine the patient's disease."},
            {"role": "assistant", "content": "<think> " + content + " </think> " + disease},
        ],
        "images": [img_path],

        "rejected_messages": [
            {"role": "user", "content": "This is a patient's chest DR image <image>. Please help me determine the patient's disease."},
            {"role": "assistant", "content": "<think> " + content + " </think> " + disease},
        ],
        "rejected_images": [cf_img_path]        
    }

    jsons.append(dialogue)

save_json(jsons, '/home/xiaoyyan/Data/code/2512-drift/data/mimic-rft-counterfact-visual-cot.json')


    

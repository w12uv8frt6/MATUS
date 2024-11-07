#Specify GPU
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import datetime
import time
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from unixcoder import UniXcoder 
import time
import chardet
import json
import json5
import argparse

import matplotlib.pyplot as plt
import ruptures as rpt
import numpy as np

#define line_information struct
class line_information:
    def __init__(self):
        self.linenumber = -1
        self.line_content = ''
        self.similarity = -1


def find_c_files_withAstPdg(directory):
    c_files = []  
    for root, dirs, files in os.walk(directory):  
        for file in files:  
            if file.endswith('.c'):  
                path_pdgdot_temp = file + ".dot"
                path_ast_temp = file + ".ast"
                if os.path.exists(os.path.join(root, file)) and os.path.exists(os.path.join(root, path_pdgdot_temp)) and os.path.exists(os.path.join(root, path_ast_temp)):
                    if (os.stat(os.path.join(root, file))).st_size > 100 and (os.stat(os.path.join(root, path_pdgdot_temp))).st_size > 100 and (os.stat(os.path.join(root, path_ast_temp))).st_size > 100:
                        c_files.append(os.path.join(root, file))  
    return c_files


def check_folder_exists(folder_path):
    return os.path.exists(folder_path) and os.path.isdir(folder_path)


def remove_duplicate_sublists(lists):
    tuples = [tuple(lst) for lst in lists]
    unique_tuples = set(tuples)
    unique_lists = [list(tpl) for tpl in unique_tuples]
    return unique_lists


if __name__ == "__main__":

    startTime=datetime.datetime.now()

    parser = argparse.ArgumentParser(description='get_rootStatement_keyVariables')

    parser.add_argument('--folder_json_path_new', type=str, default="../get_slice/results/", help='folder_json_path_new')
    parser.add_argument('--input_seed_function_filename', type=str, default="", help='input_seed_function_filename')
    parser.add_argument('--folder_log_path_new', type=str, default="../retrieve_slice/results/", help='folder_log_path_new')
    parser.add_argument('--input_seed', type=str, default="", help='input_seed')
    parser.add_argument('--model_path', type=str, default="", help='model_path')

    args = parser.parse_args()

    folder_json_path_new = args.folder_json_path_new
    input_seed_function_filename = args.input_seed_function_filename
    folder_log_path_new = args.folder_log_path_new
    input_seed = args.input_seed
    model_path = args.model_path

    if ".c" in input_seed_function_filename:
        json_name_new = "get_slice-" + input_seed_function_filename.strip()[:-2] + ".json" 
    else:
        json_name_new = "get_slice-" + input_seed_function_filename.strip() + ".json" 
    result_jsonfilepath_new = os.path.join(folder_json_path_new, json_name_new)

    if ".c" in input_seed_function_filename:
        json_name_new = "retrieve_slice-" + input_seed_function_filename.strip()[:-2] + ".json" 
    else:
        json_name_new = "retrieve_slice-" + input_seed_function_filename.strip() + ".json" 
    result_jsonfilepath_output_new = os.path.join(folder_log_path_new, json_name_new)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniXcoder(model_path)
    model.to(device)

# Encode one
    with torch.no_grad():
        tokens_ids = model.tokenize([input_seed],max_length=1023,mode="<encoder-only>")
        source_ids = torch.tensor(tokens_ids).to(device)
        tokens_embeddings,input_seed_embedding = model(source_ids)
        norm_input_embedding = torch.nn.functional.normalize(input_seed_embedding, p=2, dim=1)

        # Encode two
        with open(result_jsonfilepath_new, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        f.close()
        list_result = []
        for index, dict_temp1 in enumerate(json_data["target_functions_list"]):
            path_c_temp = dict_temp1["target_function_filepath"]
            list_temp1 = []
            for dict_temp2 in dict_temp1["list_top_statements"]:
                list_temp1 = list_temp1 + dict_temp2["list_subGraph_withlinenumber_in_statement"]
            list_temp1 = remove_duplicate_sublists(list_temp1)
            
            list_temp1_new = []
            for temp in list_temp1:
                if len(temp) == 1 and '1' in temp:
                    continue
                if len(temp) == 2 and '1' in temp: 
                    continue
                if len(temp) >= 2: 
                    if '1' in temp:
                        temp.remove('1')
                    list_temp1_new.append(temp)

            if len(list_temp1_new) > 0:
                with open(path_c_temp, 'rb') as f_temp:
                    encoding_message = chardet.detect(f_temp.read())
                f_temp.close()
                if encoding_message['encoding'] == "GB2312":
                    encoding_message['encoding'] = "GB18030"
                elif encoding_message['encoding'] == "ascii":
                    encoding_message['encoding'] = "iso8859-1"
                '''elif encoding_message['encoding'] == "Windows-1252" or encoding_message['encoding'] == "Windows-1254":
                    encoding_message['encoding'] = "utf-8"'''
                lines_c = []
                with open(path_c_temp, 'r', encoding = encoding_message['encoding']) as f:
                    lines_c = f.readlines()
                f.close()

                max_similarity = 0.0
                max_similarity_linenumberlist = []
                for temp in list_temp1_new:
                    containers = ""
                    temp = sorted(temp, key=int)
                    for temp1 in temp:
                        line_str = ""
                        line_str = lines_c[int(temp1)-1]
                        containers = containers + line_str
                    tokens_ids = model.tokenize([containers],max_length=1023,mode="<encoder-only>")
                    source_ids = torch.tensor(tokens_ids).to(device)
                    tokens_embeddings,containers_embedding = model(source_ids)
                    norm_containers_embedding = torch.nn.functional.normalize(containers_embedding, p=2, dim=1)
                    input_containers_similarity = torch.einsum("ac,bc->ab",norm_input_embedding, norm_containers_embedding)
                    if float(input_containers_similarity.item()) > max_similarity and float(input_containers_similarity.item()) > 0 and float(input_containers_similarity.item()) <= 1.0:
                        max_similarity = float(input_containers_similarity.item())
                        max_similarity_linenumberlist = temp
                    
                dict_result_temp = dict()
                if max_similarity > 0.0 and max_similarity <= 1.0:
                    dict_result_temp = {"path_c": path_c_temp, "max_score": max_similarity, "max_similarity_linenumberlist": max_similarity_linenumberlist}
                    list_result.append(dict_result_temp)


        list_result_sorted = sorted(list_result, key=lambda x: x['max_score'], reverse=True)

        result_output_json = []
        order = 1
        for item in list_result_sorted:
            if not result_output_json or item['max_score'] != result_output_json[-1]['c_files'][0]['max_score']:
                result_output_json.append({'order': order, 'c_files': []})
                order += 1
            result_output_json[-1]['c_files'].append(item)
            
            
        max_scores = [file['max_score'] for item in json_data for file in item['c_files']]
        len_max_scores = len(max_scores)
        max_scores_top = max_scores[:int(0.1 * len_max_scores)]
        max_scores_array = np.array(max_scores_top, dtype=np.float64)
        max_scores_array.shape = (len(max_scores_top),)
        algo = rpt.Dynp(model="l2", min_size=3).fit(max_scores_array) 
        result = algo.predict(n_bkps=3)
        result = result[:-1]
        top2part = int(result[1])
        
        with open(result_jsonfilepath_output_new, 'w') as f:
            json.dump(result_output_json[:top2part], f, indent=4)
        f.close()
  
        print("result_jsonfilepath_output_new: \n", result_jsonfilepath_output_new)
        print("\nretrieve_slice success.")
    
    endTime=datetime.datetime.now()
    print("start time: ", startTime)
    print("end time: ", endTime)
    diffrentTime=(endTime-startTime).seconds
    print("different time: ", diffrentTime, "s")






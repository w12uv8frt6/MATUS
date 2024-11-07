import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
import datetime
import time
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from unixcoder import UniXcoder 
import time
import chardet
import argparse

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


if __name__ == "__main__":

    startTime=datetime.datetime.now()

    parser = argparse.ArgumentParser(description='get_top_functions')
    
    parser.add_argument('--path_of_codeset', type=str, default="../datasets/linux-6.4-rc2-sepFile2function-withAstPdgEmbeddings", help='path_of_codeset')
    parser.add_argument('--model_path', type=str, default="../model/CodeXGLUE/Code-Code/Clone-detection-POJ-104/saved_models/checkpoint-best-map/pytorch_model_bin/UniXcoder-base-nine-roberta-codeClone", help='model_path')

    args = parser.parse_args()

    path_of_codeset = args.path_of_codeset
    model_path = args.model_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniXcoder(model_path)

    model.to(device)

    list_c_files = find_c_files_withAstPdg(path_of_codeset)
    list_c_files = sorted(list_c_files)

    # Encode one
    with torch.no_grad():

        #encode two
        for path_c_temp in list_c_files:
            with open(path_c_temp, 'rb') as f_temp:
                encoding_message = chardet.detect(f_temp.read())
            f_temp.close()
            if encoding_message['encoding'] == "GB2312":
                encoding_message['encoding'] = "GB18030"
            elif encoding_message['encoding'] == "ascii":
                encoding_message['encoding'] = "iso8859-1"
            '''elif encoding_message['encoding'] == "Windows-1252" or encoding_message['encoding'] == "Windows-1254":
                encoding_message['encoding'] = "utf-8"'''
            with open(path_c_temp, 'r', encoding = encoding_message['encoding']) as f:
                containers = ""
                containers = f.read()
                containers = containers.replace("\\\n", "")
                containers = containers.replace("\\n", "")
                containers = containers.replace("\n", "")
                containers = containers.replace("\\\t", " ")
                containers = containers.replace("\\t", " ")
                containers = containers.replace("\t", " ")
            f.close()

            tokens_ids = model.tokenize([containers],max_length=1023,mode="<encoder-only>")
            source_ids = torch.tensor(tokens_ids).to(device)
            tokens_embeddings,containers_embedding = model(source_ids)
            norm_containers_embedding = torch.nn.functional.normalize(containers_embedding, p=2, dim=1)

            save_path_temp = ""
            if path_c_temp.endswith(".c"):
                save_path_temp = path_c_temp[:-2] + "-FunctionEmbedding.pth"
                torch.save(norm_containers_embedding, save_path_temp)
    
    print("End Embedding.\n")
    
    endTime=datetime.datetime.now()
    print("start time: ", startTime)
    print("end time: ", endTime)
    diffrentTime=(endTime-startTime).seconds
    print("different time: ", diffrentTime, "s")






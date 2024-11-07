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

def find_c_files_withAstPdgEmbeddings(directory):
    c_files = []  
    for root, dirs, files in os.walk(directory):  
        for file in files:  
            if file.endswith('.c'):  
                path_pdgdot_temp = file + ".dot"
                path_ast_temp = file + ".ast"
                path_embedding_temp = file[:-2] + "-FunctionEmbedding.pth"
                if os.path.exists(os.path.join(root, file)) and os.path.exists(os.path.join(root, path_pdgdot_temp)) and os.path.exists(os.path.join(root, path_ast_temp)) and os.path.exists(os.path.join(root, path_embedding_temp)):
                    if (os.stat(os.path.join(root, file))).st_size > 100 and (os.stat(os.path.join(root, path_pdgdot_temp))).st_size > 100 and (os.stat(os.path.join(root, path_ast_temp))).st_size > 100 and (os.stat(os.path.join(root, path_embedding_temp))).st_size > 0:
                        c_files.append(os.path.join(root, file))  
    return c_files


def check_folder_exists(folder_path):
    return os.path.exists(folder_path) and os.path.isdir(folder_path)


if __name__ == "__main__":

    startTime=datetime.datetime.now()

    parser = argparse.ArgumentParser(description='get_top_functions')
    
    parser.add_argument('--path_of_codeset', type=str, default="", help='path_of_codeset')
    parser.add_argument('--path_of_retrieve', type=str, default="", help='path_of_retrieve')
    parser.add_argument('--input_seed_path', type=str, default="", help='input_seed_path')
    parser.add_argument('--input_seed_function_name', type=str, default="", help='input_seed_function_name')
    parser.add_argument('--top_k', type=int, default=2000, help='top_k')
    parser.add_argument('--model_path', type=str, default="", help='model_path')

    args = parser.parse_args()

    path_of_codeset = args.path_of_codeset
    path_of_retrieve = args.path_of_retrieve
    input_seed_path = args.input_seed_path
    input_seed_function_name = args.input_seed_function_name
    input_seed_path = os.path.join(input_seed_path, input_seed_function_name)
    top_k = args.top_k
    model_path = args.model_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniXcoder(model_path)
    model.to(device)

    list_c_files = find_c_files_withAstPdgEmbeddings(path_of_codeset)
    list_c_files_retrieve = find_c_files_withAstPdg(path_of_retrieve)
    list_c_files = list_c_files + list_c_files_retrieve
    list_c_files = sorted(list_c_files)

    # Encode one
    with torch.no_grad():
        with open(input_seed_path, 'rb') as f_temp:
            encoding_message = chardet.detect(f_temp.read())
        f_temp.close()
        if encoding_message['encoding'] == "GB2312":
            encoding_message['encoding'] = "GB18030"
        elif encoding_message['encoding'] == "ascii":
            encoding_message['encoding'] = "iso8859-1"
        with open(input_seed_path, 'r', encoding = encoding_message['encoding']) as f:
            input_seed = ""
            input_seed = f.read()
            input_seed = input_seed.replace("\\\n", "")
            input_seed = input_seed.replace("\\n", "")
            input_seed = input_seed.replace("\n", "")
            input_seed = input_seed.replace("\\\t", " ")
            input_seed = input_seed.replace("\\t", " ")
            input_seed = input_seed.replace("\t", " ")
        f.close()
        tokens_ids = model.tokenize([input_seed],max_length=1023,mode="<encoder-only>")
        source_ids = torch.tensor(tokens_ids).to(device)
        tokens_embeddings,input_seed_embedding = model(source_ids)
        norm_input_embedding = torch.nn.functional.normalize(input_seed_embedding, p=2, dim=1)

        str_of_file_endwith = ".c"

        list_dict = [] 
        #encode two
        list_similarity = []
        print("len(list_c_files): \n", len(list_c_files))
        for path_c_temp in list_c_files:
            path_embedding_temp = path_c_temp[:-2] + "-FunctionEmbedding.pth"

            if os.path.exists(path_embedding_temp): 
                norm_containers_embedding = torch.load(path_embedding_temp)

            else: 
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
            
            input_containers_similarity = torch.einsum("ac,bc->ab",norm_input_embedding, norm_containers_embedding)

            dict_temp = {"path_c": path_c_temp, "score": float(input_containers_similarity.item())}
            list_dict.append(dict_temp)

    list_dict_desc = sorted(list_dict, key=lambda x: x['score'], reverse=True) 

    folder_log_path = "./get_top_functions/results/"
    if ".c" in input_seed_function_name:
        log_name = "getTopFunctions-" + input_seed_function_name.strip()[:-2] + ".log"
    else:
        log_name = "getTopFunctions-" + input_seed_function_name.strip() + ".log"
    i = 0
    if check_folder_exists(folder_log_path):
        log_path = os.path.join(folder_log_path, log_name)
        with open(log_path, 'w') as f:
            for line in list_dict_desc:
                f.write("order: " + str(i+1) + "\nscore: " + str(list_dict_desc[i]["score"]) + "\npath_c: " + str(list_dict_desc[i]["path_c"]) + "\n")
                i += 1
                if i >= top_k:
                    break
        f.close()
    else:
        print("\nfolder not exist!\n", folder_log_path)
    
    print("\nfolder_log_path: ", folder_log_path)
    print("\nlog_name: ", log_name)
    print("\ninput_seed_path: ", input_seed_path)
    
    endTime=datetime.datetime.now()
    print("start time: ", startTime)
    print("end time: ", endTime)
    diffrentTime=(endTime-startTime).seconds
    print("different time: ", diffrentTime, "s")






#Specify GPU
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
import re
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
import math

from tree_sitter import Language, Parser  
from graphviz import Digraph 


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


def remove_duplicates(strings):
    seen = set()
    return [x for x in strings if not (x in seen or seen.add(x))]


def ast_mapping_function(root_node):
    seq = []  
    name = root_node.type  

    if len(root_node.children) == 0:
        seq.append(root_node.text.decode('utf8') if root_node.text else '')
    else:
        seq.append(f"<{name},left>")
        for child in root_node.children:
            seq.extend(ast_mapping_function(child))
        seq.append(f"<{name},right>")

    return seq


if __name__ == "__main__":

    startTime=datetime.datetime.now()

    parser = argparse.ArgumentParser(description='get_rootStatement_keyVariables')

    parser.add_argument('--path_of_codeset', type=str, default="../datasets/linux-6.4-rc2-sepFile2function-withAstPdgEmbeddings", help='path_of_codeset')
    parser.add_argument('--model_path', type=str, default="", help='model_path')

    args = parser.parse_args()

    path_of_codeset = args.path_of_codeset
    model_path = args.model_path

    list_path_c = []
    list_path_c = find_c_files_withAstPdgEmbeddings(path_of_codeset)
    list_path_c = sorted(list_path_c)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:\n", device)
    model = UniXcoder(model_path)
    model.to(device)

    # Encode one
    with torch.no_grad():
        
        for order, path_c_temp in enumerate(list_path_c):
            lines_c = []
            #Encode two
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
                lines_c = f.readlines()
            f.close()

            for linenumber, line_temp in enumerate(lines_c):
                containers = ""
                containers = line_temp.strip()
                if len(containers) > 0:
                    C_LANGUAGE = Language('./build/my-languages.so', 'c')
                    c_parser = Parser()
                    c_parser.set_language(C_LANGUAGE)

                    tree = c_parser.parse(bytes(containers, "utf8"))
                    root_node = tree.root_node 
                    flattened_ast_sequence = str(ast_mapping_function(root_node))
                    tokens_ids = model.tokenize([flattened_ast_sequence],max_length=1023,mode="<encoder-only>")
                    source_ids = torch.tensor(tokens_ids).to(device)
                    tokens_embeddings,containers_embedding = model(source_ids)
                    norm_containers_embedding = torch.nn.functional.normalize(containers_embedding, p=2, dim=1)

                    save_path_temp = ""
                    if path_c_temp.endswith(".c"):
                        save_path_temp = path_c_temp[:-2] + "-LineASTEmbedding-" + str(linenumber + 1) + ".pth"
                        torch.save(norm_containers_embedding, save_path_temp)

    print("All c files LineAST embedding finished.\n")

    endTime=datetime.datetime.now()
    print("start time: ", startTime)
    print("end time: ", endTime)
    diffrentTime=(endTime-startTime).seconds
    print("different time: ", diffrentTime, "s")






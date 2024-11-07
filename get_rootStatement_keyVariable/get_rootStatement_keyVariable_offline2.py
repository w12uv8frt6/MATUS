#Specify GPU
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

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


def find_c_files(directory):
    c_files = []  
    for root, dirs, files in os.walk(directory):  
        for file in files:  
            if file.endswith('.c'):  
                c_files.append(os.path.join(root, file))  
    return c_files


def check_folder_exists(folder_path):
    return os.path.exists(folder_path) and os.path.isdir(folder_path)


def remove_duplicates(strings):
    seen = set()
    return [x for x in strings if not (x in seen or seen.add(x))]


def ast_analyse(path_c_temp):
    dict_ast_analyse = dict()
    lines_c = []
    path_ast_temp = path_c_temp.strip() + ".ast"

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
    count_lines = len(lines_c)
    for i in range(count_lines):
        dict_ast_analyse[str(i+1)] = [int(i+1)]

    with open(path_ast_temp, 'rb') as f_temp:
        encoding_message = chardet.detect(f_temp.read())
    f_temp.close()
    if encoding_message['encoding'] == "GB2312":
        encoding_message['encoding'] = "GB18030"
    elif encoding_message['encoding'] == "ascii":
        encoding_message['encoding'] = "iso8859-1"
    '''elif encoding_message['encoding'] == "Windows-1252" or encoding_message['encoding'] == "Windows-1254":
        encoding_message['encoding'] = "utf-8"'''
    with open(path_ast_temp, 'r', encoding = encoding_message['encoding']) as f:
        lines_ast = f.readlines()
    f.close()

    current_linenumber = 0
    Statement_linenumber = 0
    Label_linenumber = 0
    for line_ast_temp in lines_ast:
        line_ast_temp = line_ast_temp.strip()
        if ":FunctionDef:" in line_ast_temp:
            continue
        if ":CompoundStatement:" in line_ast_temp:
            continue
        if "ElseStatement" in line_ast_temp:
            continue
        if ":Label:" in line_ast_temp:
            if not (line_ast_temp.startswith("None") or line_ast_temp.startswith(":")):
                parts = line_ast_temp.split(":", 2)
                if len(parts) == 3 and parts[0].strip().isdigit():
                    Label_linenumber = int(parts[0].strip())
            continue
        if not (line_ast_temp.startswith("None") or line_ast_temp.startswith(":")):
            parts = line_ast_temp.split(":", 2)
            if len(parts) == 3 and parts[0].strip().isdigit() and int(parts[0].strip()) > current_linenumber:
                current_linenumber = int(parts[0].strip())
        if current_linenumber == Label_linenumber:
            continue
        if "Statement:" in line_ast_temp:
            if not (line_ast_temp.startswith("None") or line_ast_temp.startswith(":")):
                parts = line_ast_temp.split(":", 2)
                if len(parts) == 3 and "Statement" in parts[1] and parts[0].strip().isdigit():
                    Statement_linenumber = int(parts[0].strip())
        if current_linenumber == Statement_linenumber:
            continue
        if current_linenumber > Statement_linenumber:
            if Statement_linenumber > 0 and (current_linenumber - 1) in dict_ast_analyse[str(Statement_linenumber)] and current_linenumber not in dict_ast_analyse[str(Statement_linenumber)]:
                dict_ast_analyse[str(Statement_linenumber)].append(current_linenumber)
    for key, value in dict_ast_analyse.items():
        if len(value) > 1:
            for value_temp in value:
                dict_ast_analyse[str(value_temp)] = value

    return dict_ast_analyse


if __name__ == "__main__":

    startTime=datetime.datetime.now()

    parser = argparse.ArgumentParser(description='get_rootStatement_keyVariables')

    parser.add_argument('--folder_log_path', type=str, default="../get_top_functions/results/", help='folder_log_path')
    parser.add_argument('--folder_json_path', type=str, default="../get_rootStatement_keyVariable/results/", help='folder_json_path')

    args = parser.parse_args()

    folder_log_path = args.folder_log_path
    folder_json_path = args.folder_json_path

    log_name = "getTopFunctions-all.log" 
    json_name = "get_rootStatement_keyVariable-all.json" 

    log_path = os.path.join(folder_log_path, log_name)
    result_jsonfilepath = os.path.join(folder_json_path, json_name)

    lines_log = []
    with open(log_path, 'r', encoding = 'utf-8') as f:
        lines_log = f.readlines()
    f.close()
    list_path_topk = []
    for line_log_temp in lines_log:
        line_log_temp = line_log_temp.strip()
        if "order: " in line_log_temp:
            match_line_log_temp = re.search(r'\d+', line_log_temp)
            if match_line_log_temp:
                order = int(match_line_log_temp.group())
                continue
            else:
                print("No number found")
                print("time.sleep(100000)")
                time.sleep(100000)
        if "path_c: " in line_log_temp:
            match_line_log_temp = re.search(r':\s*(.*)', line_log_temp)
            if match_line_log_temp:
                path = match_line_log_temp.group(1)
                list_path_topk.append(path)
                continue
            else:
                print("No path found")
                print("time.sleep(100000)")
                time.sleep(100000)
    
    top_k = order
    if len(list_path_topk) != top_k:
        print("top_k: ", top_k)
        print("len(list_path_topk) != top_k")
        print("time.sleep(100000)")
        time.sleep(100000)

    str_of_file_endwith = ".c"
    dict_seed_target = dict()
    target_functions_list = []
    for order, path_c_temp in enumerate(list_path_topk):
        dict_target_function = dict()
        dict_target_function["order"] = int(order+1)
        dict_target_function["target_function_filepath"] = str(path_c_temp)
        lines_c = []
        list_dict = []

        #Encode two
        list_similarity = []
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

        top_k_statement = int(len(lines_c))

        for linenumber, line_temp in enumerate(lines_c):
            containers = ""
            containers = line_temp.strip()
            if len(containers) > 0:
                dict_temp = {"linenumber": int(linenumber+1), "line": str(containers)}
                list_dict.append(dict_temp)
        
        if len(list_dict) <= 0:
            continue
        else:
            dict_ast_analyse = ast_analyse(path_c_temp)
            list_top_statements = []
            for dict_temp in list_dict:
                list_linenumber = dict_ast_analyse[str(dict_temp["linenumber"])] 
                path_ast_temp = path_c_temp + ".ast"
                with open(path_ast_temp, 'rb') as f_temp:
                    encoding_message = chardet.detect(f_temp.read())
                f_temp.close()
                if encoding_message['encoding'] == "GB2312":
                    encoding_message['encoding'] = "GB18030"
                elif encoding_message['encoding'] == "ascii":
                    encoding_message['encoding'] = "iso8859-1"
                '''elif encoding_message['encoding'] == "Windows-1252" or encoding_message['encoding'] == "Windows-1254":
                    encoding_message['encoding'] = "utf-8"'''
                with open(path_ast_temp, 'r', encoding = encoding_message['encoding']) as f:
                    lines_ast = f.readlines()
                f.close()
                list_potential_variables = []
                for linenumber_temp in list_linenumber:
                    Callee_str = ""
                    for line_ast_temp in lines_ast:
                        line_ast_temp = line_ast_temp.strip()
                        if not line_ast_temp.startswith("None") and not line_ast_temp.startswith(":") and line_ast_temp.count(':') >= 2:
                            parts = line_ast_temp.split(":", 2)
                            if len(parts) == 3 and parts[0].strip().isdigit():
                                if parts[1].strip() == "Callee":
                                    Callee_str = str(parts[2].strip())
                                if parts[1].strip() == "UnaryExpression" and parts[2].strip() != Callee_str and not str(parts[2].strip()).isupper() and linenumber_temp == int(parts[0].strip()):
                                    list_potential_variables.append(str(parts[2].strip()))
                                if parts[1].strip() == "Identifier" and parts[2].strip() != Callee_str and not str(parts[2].strip()).isupper() and linenumber_temp == int(parts[0].strip()):
                                        list_potential_variables.append(str(parts[2].strip()))
                                if parts[1].strip() == "ArrayIndexing" and parts[2].strip() != Callee_str and not str(parts[2].strip()).isupper() and linenumber_temp == int(parts[0].strip()):
                                    list_potential_variables.append(str(parts[2].strip()))
                                if parts[1].strip() == "MemberAccess" and parts[2].strip() != Callee_str and not str(parts[2].strip()).isupper() and linenumber_temp == int(parts[0].strip()):
                                    list_potential_variables.append(str(parts[2].strip()))
                        else:
                            if line_ast_temp.startswith("None:Callee:"):
                                parts = line_ast_temp.split(":", 2)
                                if len(parts) == 3 and parts[1].strip() == "Callee" and parts[0].strip() == "None":
                                    Callee_str = str(parts[2].strip())
                                    continue
                list_potential_variables = remove_duplicates(list_potential_variables)
                if len(list_potential_variables) <= 0:
                    continue

                dict_top_statement = dict()
                dict_top_statement["statement_linenumber"] = int(dict_temp["linenumber"])
                dict_top_statement["only_one_statement"] = True if len(list_linenumber) == 1 else False
                dict_top_statement["statement_linenumber_start"] = min(list_linenumber)
                dict_top_statement["statement_linenumber_stop"] = max(list_linenumber)
                dict_top_statement["list_potential_variables_in_statement"] = list_potential_variables
                if int(dict_temp["linenumber"]) == 1: 
                    continue 
                list_top_statements.append(dict_top_statement)
        
        dict_target_function["list_top_statements"] = list_top_statements
        target_functions_list.append(dict_target_function)
    
    dict_seed_target["target_functions_list"] = target_functions_list

    json_data = json.dumps(dict_seed_target, indent=4, ensure_ascii=False)
    with open(result_jsonfilepath, "w", encoding="utf-8") as f:
        f.write(json_data)
    f.close()
    print("result_jsonfilepath: \n", result_jsonfilepath)
    print("get_rootStatement_keyVariable.py success.")


    endTime=datetime.datetime.now()
    print("start time: ", startTime)
    print("end time: ", endTime)
    diffrentTime=(endTime-startTime).seconds
    print("different time: ", diffrentTime, "s")






#Specify GPU
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from unixcoder import UniXcoder
import shutil
import re
import sys
import time
import datetime
import networkx as nx
import json
import json5
import random
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import string
import chardet
import matplotlib.pyplot as plt
import argparse
import html
import math


def check_rootstatement_in_subgraph(lst, start, stop):
    int_list = [int(item) for item in lst]

    for num in int_list:
        if start <= num <= stop:
            return True
    return False


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


def find_c_files(directory):  
    c_files = []  
    for root, dirs, files in os.walk(directory):  
        for file in files:  
            if file.endswith('.c'):  
                c_files.append(os.path.join(root, file))  
    return c_files


def remove_duplicates(lst):  
    unique_list = []
    for item in lst:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list


def delete_selfCircle_edge(list_edge): 
    new_list_edge = []
    for edge_temp in list_edge:
        if edge_temp[0] == edge_temp[1]:
            continue
        else:
            new_list_edge.append(edge_temp)
    return new_list_edge


def getPdgGraph_fromdot(path_pdgdot):
    G = nx.MultiDiGraph()
    dict_nodeid_lineNum = {}
    func_name = ""
    list_G_add_edges = []
    list_G_add_nodes = []

    with open(path_pdgdot, 'r') as f:
        lines = f.readlines()
    f.close()

    for line in lines:
        if "</SUB>> ]" in line and "[label = <" in line and "<SUB>" in line:
            node_id = line.split("[label = ")[0].split("\"")[1]
            index_sub1 = line.find("<SUB>")
            index_sub2 = line.find("</SUB>")
            node_lineNum = line[index_sub1 + 5:index_sub2]
            dict_nodeid_lineNum[node_id] = node_lineNum 
            list_G_add_nodes.append(node_lineNum) 
        
        elif "digraph \"" in line and "\" {" in line:
            start_index = line.find('"') + 1  
            end_index = line.find('"', start_index)  
            func_name = line[start_index:end_index]  
    
    for line in lines:
        if "[ label = \"DDG:" in line:
            start_index = line.find('"') + 1  
            end_index = line.find('"', start_index)  
            node_id1 = line[start_index:end_index]  
            start_index = end_index + 1  
            start_index = line.find('"', start_index) + 1
            end_index = line.find('"', start_index)  
            node_id2 = line[start_index:end_index]

            start_index = line.find('label = "DDG: ') + 14
            end_index = line.find('"]', start_index)  
            edge_DDG = line[start_index:end_index]  

            edge_DDG = edge_DDG.strip()
            edge_DDG = html.unescape(edge_DDG) 
            if edge_DDG.startswith("!") or edge_DDG.startswith("*") or edge_DDG.startswith("&"):
                edge_DDG = edge_DDG[1:]
            if edge_DDG.startswith("**"):
                edge_DDG = edge_DDG[2:]

            if dict_nodeid_lineNum.get(node_id1) and dict_nodeid_lineNum.get(node_id2): 
                tuple_temp_edge = (dict_nodeid_lineNum[node_id1], str("DDG: " + edge_DDG), dict_nodeid_lineNum[node_id2])
                tuple_temp_edge = (dict_nodeid_lineNum[node_id1], dict_nodeid_lineNum[node_id2], {'edge_info': str("DDG: " + edge_DDG)})
                list_G_add_edges.append(tuple_temp_edge)

        elif "[ label = \"CDG:" in line:
            start_index = line.find('"') + 1  
            end_index = line.find('"', start_index)  
            node_id1 = line[start_index:end_index]  
            start_index = end_index + 1
            start_index = line.find('"', start_index) + 1
            end_index = line.find('"', start_index)  
            node_id2 = line[start_index:end_index]

            if dict_nodeid_lineNum.get(node_id1) and dict_nodeid_lineNum.get(node_id2):
                tuple_temp_edge = (dict_nodeid_lineNum[node_id1], dict_nodeid_lineNum[node_id2], {'edge_info': str("CDG")})
                list_G_add_edges.append(tuple_temp_edge)

    list_G_add_edges = delete_selfCircle_edge(list_G_add_edges)
    list_G_add_edges = remove_duplicates(list_G_add_edges)
    list_G_add_nodes = remove_duplicates(list_G_add_nodes)

    if len(list_G_add_nodes) > 0:
        G.add_nodes_from(list_G_add_nodes)
    if len(list_G_add_edges) > 0:
        G.add_edges_from(list_G_add_edges)

    return G


def is_variable(s): 
    if len(s) == 0:
        return False
    
    keywords = ['auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do', 'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if', 'int', 'long', 'register', 'return', 'short', 'signed', 'sizeof', 'static', 'struct', 'switch', 'typedef', 'union', 'unsigned', 'void', 'volatile', 'while', 'class', 'constructor', 'destructor', 'do', 'while', 'for', 'if', 'else', 'switch', 'case', 'default', 'typeid', 'namespace', 'using', 'inline', 'virtual', 'override', 'abstract', 'concrete', 'bool', 'true', 'false']  
    for keyword in keywords:
        if s == keyword:
            return False

    if s[0].isdigit():
        return False

    if s.isupper():
        return False

    if ">" in s and not "->" in s:
        return False
    if "[" in s and not "]" in s:
        return False
    if "]" in s and not "[" in s:
        return False
    if "\011" in s or "\012" in s or "(" in s or ")" in s or "<" in s or "*" in s or "!" in s or "&" in s or "#" in s or s.startswith("struct "):
        return False

    return True


def get_DDGvariable(pdgGraph_temp): 
    list_DDGvariable = []
    for edge_temp in pdgGraph_temp.edges():
        list_relation_temp = pdgGraph_temp[edge_temp[0]][edge_temp[1]]
        for key, relation_temp in list_relation_temp.items():
            if "DDG: " in relation_temp['edge_info']:
                variable_temp = relation_temp['edge_info'][5:]
                
                if not is_variable(variable_temp):
                    continue
                else:
                    list_DDGvariable.append(variable_temp)
    list_DDGvariable = remove_duplicates(list_DDGvariable)
    return list_DDGvariable    


def get_subGraph(list_DDGvariable, pdgGraph_temp, len_edges, path_ast):
    list_subGraph_withLineNumber = []
    list_subGraph = []
    
    for DDGvariable_temp in list_DDGvariable:
        subGraph_lineNumber_temp = {}
        subGraph_temp = {}
        list_subGraphTemp_withLineNumber = []
        subgraph_edges = []
        list_DDGvariable_temp_lineNumber = []

        with open(path_ast, 'rb') as f_temp:
            encoding_message = chardet.detect(f_temp.read())
        f_temp.close()
        if encoding_message['encoding'] == "GB2312":
            encoding_message['encoding'] = "GB18030"
        elif encoding_message['encoding'] == "ascii":
            encoding_message['encoding'] = "iso8859-1"
        '''elif encoding_message['encoding'] == "Windows-1252" or encoding_message['encoding'] == "Windows-1254":
            encoding_message['encoding'] = "utf-8"'''
        with open(path_ast, 'r', encoding = encoding_message['encoding']) as f_ast:
            lines = f_ast.readlines()
            for line in lines:
                line = line.strip()
                if ":Identifier:" in line:
                    find_Identifier_temp = line[(line.find(":Identifier:") + 12):].strip()
                    find_lineNumber_temp = line[:(line.find(":Identifier:"))].strip()
                    if DDGvariable_temp == find_Identifier_temp:
                        list_DDGvariable_temp_lineNumber.append(DDGvariable_temp)
        f_ast.close()
        list_DDGvariable_temp_lineNumber = remove_duplicates(list_DDGvariable_temp_lineNumber)

        for from_node, to_node, edge_attributes in pdgGraph_temp.edges(data=True): 
            flag1 = edge_attributes['edge_info'][5:] == DDGvariable_temp and from_node != to_node
            flag2 = DDGvariable_temp in edge_attributes['edge_info'][5:] and from_node != to_node and from_node in list_DDGvariable_temp_lineNumber and to_node in list_DDGvariable_temp_lineNumber
            if flag1 or flag2:
                subgraph_edges.append((from_node,to_node,edge_attributes))

        subgraph_edges = remove_duplicates(subgraph_edges) 
        subgraph = nx.MultiDiGraph()
        subgraph.add_edges_from(subgraph_edges)
        
        if len(subgraph.edges(data=True)) >= len_edges:

            subgraph_undirected = nx.Graph(subgraph)
            list_graphs = list(nx.connected_components(subgraph_undirected))
            if len(list_graphs) > 1:
                max_subgraph_nodeset = max(list_graphs)
                set_nodes_subgraphs = set(subgraph.nodes())
                set_nodes_needDelete = set_nodes_subgraphs - max_subgraph_nodeset
                for node_temp in set_nodes_needDelete:
                    subgraph.remove_edges_from([(u, v) for u, v in subgraph.edges() if u == node_temp or v == node_temp])  
                    subgraph.remove_node(node_temp)

            for edge_temp in subgraph.edges(data=True):
                list_subGraphTemp_withLineNumber.append(edge_temp[0])
                list_subGraphTemp_withLineNumber.append(edge_temp[1])
            list_subGraphTemp_withLineNumber = remove_duplicates(list_subGraphTemp_withLineNumber) 
            list_subGraphTemp_withLineNumber.sort() 
            subGraph_lineNumber_temp = {DDGvariable_temp: list_subGraphTemp_withLineNumber}
            subGraph_temp = {DDGvariable_temp: subgraph}
            list_subGraph_withLineNumber.append(subGraph_lineNumber_temp)
            list_subGraph.append(subGraph_temp)
            
    return list_subGraph_withLineNumber, list_subGraph


def delete_tab(list_text):
    lines = list_text
    if lines[0].strip() == "":
        lines = lines[1:]
    line0 = lines[0].rstrip()
    line0_start = line0.split(":")[0].rstrip()
    tab_count_line0 = line0_start.count("\t")
    lines_new = []
    for index, line in enumerate(lines):
        tab_count_temp = line.rstrip().split(":")[0].rstrip().count("\t")
        if tab_count_temp < tab_count_line0:
            continue
        else:
            new_tab_count_temp = tab_count_temp - tab_count_line0
            new_line = int(new_tab_count_temp) * str("\t") + line.strip() + "\n"
            lines_new.append(str(new_line))

    new_text = ''.join(lines_new)
    return lines_new


def parse_text_to_json(list_text):

    lines = delete_tab(list_text)
    tree = {'name': 'root', 'children': []}
    parent_stack = [tree]

    for line in lines:
        level = 0
        while line[level] == '\t':
            level += 1
        name = line[level:].strip()

        while level < len(parent_stack) - 1:
            parent_stack.pop()

        new_node = {'name': name, 'children': []}
        parent_stack[-1]['children'].append(new_node)
        parent_stack.append(new_node)

    return json.dumps(tree, indent=2)


def find_node(tree, target_name):
    if target_name in tree['name']:
        return tree
    for child in tree.get('children', []):
        result = find_node(child, target_name)
        if result:
            return result
    return None


def get_names(tree):
    names = [tree["name"]]
    for child in tree.get("children", []):
        names.extend(get_names(child))
    return names


def replace_one_at_a_time(input_string, variable_temp, replacement):
    variable_temp = re.escape(variable_temp)
    matches = list(re.finditer(variable_temp, input_string))

    input_string_replaced = []
    if matches:
        for index, match_temp in enumerate(matches):
            start, end = match_temp.span()
            input_string_replaced.append(input_string[:start] + replacement + input_string[end:])
    
    return input_string_replaced


def generate_subGraph_data(path_pdgdot, path_ast, path_c, len_edges, list_top_statements, norm_mask_embedding_reshaped_1, model, threshold_getKeyVariable, json_data_all):
    target_file = {}
    for file_temp in json_data["target_functions_list"]: 
        if path_c.strip() in file_temp["target_function_filepath"]:
            target_file = file_temp
            break

    if target_file != {}:
        list_top_statements_new = []
        for statement_temp in list_top_statements:
            statement = dict_top_statement["statement"]
            statement_linenumber = dict_top_statement["statement_linenumber"]
            statement_similarity = dict_top_statement["statement_similarity"]
            only_one_statement = dict_top_statement["only_one_statement"]
            statement_linenumber_start = dict_top_statement["statement_linenumber_start"]
            statement_linenumber_stop = dict_top_statement["statement_linenumber_stop"]
            list_potential_variables_in_statement = dict_top_statement["list_potential_variables_in_statement"]


            target_statement = {}
            for statement_pregenerated in target_file['list_top_statements']:
                if statement_pregenerated["statement_linenumber"] == statement_temp["statement_linenumber"]:
                    target_statement = statement_pregenerated
                    break
            if target_statement == {}:
                continue
            
            list_of_dict_variable = target_statement["list_of_dict_variable"]
            list_of_dict_variable_2 = []
            for dict_temp_variable in list_of_dict_variable:
                index_for_variableMask = dict_temp_variable["index_for_variableMask"]
                variable_temp = dict_temp_variable["variable_temp"]
                index = dict_temp_variable["index"]
                path_embedding_temp =  path_c[:-2] + "____linenumber" + str(target_statement["statement_linenumber"]) + "____index_for_variableMask_" + str(index_for_variableMask) + "____maskEmbedding_" + str(index) + ".pth"
                norm_mask_embedding_reshaped_2 = torch.load(path_embedding_temp)

                input_containers_similarity = torch.einsum("ac,bc->ab",norm_mask_embedding_reshaped_1, norm_mask_embedding_reshaped_2)
                input_containers_similarity = float(input_containers_similarity.item())
                dict_temp = dict()
                dict_temp["variable_temp"] = variable_temp
                dict_temp["input_containers_similarity"] = input_containers_similarity
                list_of_dict_variable_2.append(dict_temp)

            list_subGraph_withlinenumber_in_statement = []
            if len(list_of_dict_variable_2) > 0:
                list_of_dict_variable_sorted = sorted(list_of_dict_variable_2, key=lambda x: x["input_containers_similarity"], reverse=True)
                index_temp = round(len(list_of_dict_variable_sorted) * float(threshold_getKeyVariable))
                if index_temp <= 1: 
                    index_temp = 1
                if index_temp >= len(list_of_dict_variable_sorted):
                    index_temp = len(list_of_dict_variable_sorted)
                top_list_of_dict_variable_sorted = list_of_dict_variable_sorted[:index_temp]
                new_common_elements_list = [item["variable_temp"] for item in top_list_of_dict_variable_sorted]
                new_common_elements_list = remove_duplicates(new_common_elements_list) #去重
                
                if len(new_common_elements_list) > 0:
                    for common_element_temp in new_common_elements_list:
                        subGraph_withlinenumber_temp = []
                        for subGraph_withlinenumber_in_statement_temp in target_statement["list_subGraph_withlinenumber_in_statement"]:
                            if common_element_temp == subGraph_withlinenumber_in_statement_temp["variable_name"]:
                                subGraph_withlinenumber_temp = subGraph_withlinenumber_in_statement_temp["subGraph_withlinenumber"]
                                break
                        if len(subGraph_withlinenumber_temp) == 0:
                            continue
                        if check_rootstatement_in_subgraph(subGraph_withlinenumber_temp, statement_linenumber_start, statement_linenumber_stop): 
                            list_subGraph_withlinenumber_in_statement.append(subGraph_withlinenumber_temp)

                list_subGraph_withlinenumber_in_statement = remove_duplicates(list_subGraph_withlinenumber_in_statement)
                if len(list_subGraph_withlinenumber_in_statement) > 0:
                    dict_temp = dict()
                    dict_temp["statement"] = statement
                    dict_temp["statement_linenumber"] = statement_linenumber
                    dict_temp["statement_similarity"] = statement_similarity
                    dict_temp["only_one_statement"] = only_one_statement
                    dict_temp["statement_linenumber_start"] = statement_linenumber_start
                    dict_temp["statement_linenumber_stop"] = statement_linenumber_stop
                    dict_temp["list_potential_variables_in_statement"] = list_potential_variables_in_statement
                    dict_temp["new_common_elements_list"] = new_common_elements_list 
                    dict_temp["list_subGraph_withlinenumber_in_statement"] = list_subGraph_withlinenumber_in_statement
                    list_top_statements_new.append(dict_temp)

        return list_top_statements_new

    pdgGraph_temp = getPdgGraph_fromdot(path_pdgdot)

    list_DDGvariable = get_DDGvariable(pdgGraph_temp)

    list_subGraph_withLineNumber, list_subGraph = get_subGraph(list_DDGvariable, pdgGraph_temp, len_edges, path_ast)

    list_top_statements_new  = []
    if len(list_top_statements) > 0:
        for dict_top_statement in list_top_statements:
            dict_ast_analyse = ast_analyse(path_c) 
            statement = dict_top_statement["statement"]
            statement_linenumber = dict_top_statement["statement_linenumber"]
            statement_similarity = dict_top_statement["statement_similarity"]
            only_one_statement = dict_top_statement["only_one_statement"]
            statement_linenumber_start = dict_top_statement["statement_linenumber_start"]
            statement_linenumber_stop = dict_top_statement["statement_linenumber_stop"]
            list_potential_variables_in_statement = dict_top_statement["list_potential_variables_in_statement"]
            new_append_list = []
            for temp in list_potential_variables_in_statement:
                if "->" in temp:
                    parts = temp.split("->")
                    len_parts = len(parts)
                    for i in range(len_parts):
                        variable_temp = ""
                        for j in range(i+1):
                            if j == 0:
                                variable_temp = parts[0].strip()
                            else:
                                variable_temp = variable_temp + "->" + parts[j].strip()
                        new_append_list.append(variable_temp)
            list_potential_variables_in_statement = list_potential_variables_in_statement + new_append_list
            list_potential_variables_in_statement = remove_duplicates(list_potential_variables_in_statement)
            list_subGraph_withlinenumber_in_statement = [] 

            list_variable_from_list_subGraph = [list(d.keys())[0] for d in list_subGraph]
            set1 = set(list_potential_variables_in_statement)
            set2 = set(list_variable_from_list_subGraph)
            common_elements = set1.intersection(set2)
            common_elements_list = list(common_elements)
            if len(common_elements_list) > 0:
                if len(common_elements_list) <= 2:
                    new_common_elements_list = common_elements_list
                else:
                    if not only_one_statement: 
                        with open(path_c, 'rb') as f_temp:
                            encoding_message = chardet.detect(f_temp.read())
                        f_temp.close()
                        if encoding_message['encoding'] == "GB2312":
                            encoding_message['encoding'] = "GB18030"
                        elif encoding_message['encoding'] == "ascii":
                            encoding_message['encoding'] = "iso8859-1"
                        '''elif encoding_message['encoding'] == "Windows-1252" or encoding_message['encoding'] == "Windows-1254":
                            encoding_message['encoding'] = "utf-8"'''
                        lines_c = []
                        with open(path_c, 'r', encoding = encoding_message['encoding']) as f:
                            lines_c = f.readlines()
                        f.close()
                        lines_c_new = []
                        for line_c_temp in lines_c:
                            lines_c_new.append(line_c_temp.strip())
                        statement_concatenate = ' '.join(lines_c_new[statement_linenumber_start-1:statement_linenumber_stop])
                    else:
                        statement_concatenate = statement
                    list_of_dict_variable = []
                    for variable_temp in common_elements_list:
                        list_statement_concatenate_with_mask = replace_one_at_a_time(statement_concatenate, variable_temp, "<mask>")
                        if len(list_statement_concatenate_with_mask) > 0:
                            for index, statement_concatenate_with_mask in enumerate(list_statement_concatenate_with_mask):
                                encoded_input_2 = model.tokenizer(statement_concatenate_with_mask, return_tensors='pt')
                                tokens_ids = model.tokenize([statement_concatenate_with_mask],max_length=1023,mode="<encoder-only>")
                                source_ids = torch.tensor(tokens_ids).to(device)
                                tokens_embeddings,input_seed_embedding = model(source_ids)
                                mask_token_id = model.tokenizer.convert_tokens_to_ids("<mask>")
                                mask_token_position = encoded_input_2['input_ids'].tolist()[0].index(mask_token_id)
                                if mask_token_position < tokens_embeddings.shape[1]: 
                                    mask_embedding = tokens_embeddings[0, mask_token_position, :]
                                else:
                                    continue
                                mask_embedding_reshaped = mask_embedding.unsqueeze(0)
                                norm_mask_embedding_reshaped_2 = torch.nn.functional.normalize(mask_embedding_reshaped, p=2, dim=1)

                                input_containers_similarity = torch.einsum("ac,bc->ab",norm_mask_embedding_reshaped_1, norm_mask_embedding_reshaped_2)
                                input_containers_similarity = float(input_containers_similarity.item())
                                dict_temp = dict()
                                dict_temp["variable_temp"] = variable_temp
                                dict_temp["input_containers_similarity"] = input_containers_similarity
                                list_of_dict_variable.append(dict_temp)
                    if len(list_of_dict_variable) > 0:
                        list_of_dict_variable_sorted = sorted(list_of_dict_variable, key=lambda x: x["input_containers_similarity"], reverse=True)
                        index_temp = round(len(list_of_dict_variable_sorted) * float(threshold_getKeyVariable)) 
                        if index_temp <= 1: 
                            index_temp = 1
                        if index_temp >= len(list_of_dict_variable_sorted):
                            index_temp = len(list_of_dict_variable_sorted)
                        top_list_of_dict_variable_sorted = list_of_dict_variable_sorted[:index_temp]
                        new_common_elements_list = [item["variable_temp"] for item in top_list_of_dict_variable_sorted]
                        new_common_elements_list = remove_duplicates(new_common_elements_list) #去重
                        if len(new_common_elements_list) > len(common_elements_list):
                            print("len(new_common_elements_list) > len(common_elements_list)")
                            print("invalid!!!!!")
                            time.sleep(100000)
                    else:
                        new_common_elements_list = common_elements_list

                for common_elements_temp in new_common_elements_list:
                    subGraph_withlinenumber_original = []
                    subGraph_withlinenumber = next((dct[str(common_elements_temp)] for dct in list_subGraph_withLineNumber if str(common_elements_temp) in dct), None)
                    if subGraph_withlinenumber is None:
                        continue
                    else:
                        list_line_temp_new = []
                        for line_temp in subGraph_withlinenumber: 
                            list_line_temp_new = list_line_temp_new + dict_ast_analyse[str(line_temp).strip()]
                        list_line_temp_new = [str(item) for item in list_line_temp_new]
                        subGraph_withlinenumber = subGraph_withlinenumber + list_line_temp_new
                        subGraph_withlinenumber = remove_duplicates(subGraph_withlinenumber)
                        subGraph_withlinenumber = sorted(subGraph_withlinenumber, key=int) 

                        flag_keyvariable_left_or_right = 0 
                        for linenumber_temp in subGraph_withlinenumber:
                            linenumber_temp = int(linenumber_temp)
                            if int(linenumber_temp) == 1: 
                                continue
                            value_temp = dict_ast_analyse[str(linenumber_temp)]
                            if len(value_temp) != 1: 
                                continue
                            else:
                                lineindex_start = -1
                                lineindex_end = -1
                                with open(path_ast, 'rb') as f_temp:
                                    encoding_message = chardet.detect(f_temp.read())
                                f_temp.close()
                                if encoding_message['encoding'] == "GB2312":
                                    encoding_message['encoding'] = "GB18030"
                                elif encoding_message['encoding'] == "ascii":
                                    encoding_message['encoding'] = "iso8859-1"
                                '''elif encoding_message['encoding'] == "Windows-1252" or encoding_message['encoding'] == "Windows-1254":
                                    encoding_message['encoding'] = "utf-8"'''
                                with open(path_ast, 'r', encoding = encoding_message['encoding']) as f_ast:
                                    lines_ast = f_ast.readlines()
                                    for index, line_ast_temp in enumerate(lines_ast):
                                        line_ast_temp_strip = line_ast_temp.strip()
                                        if not (line_ast_temp_strip.startswith("None") or line_ast_temp_strip.startswith(":")):
                                            parts = line_ast_temp.split(":", 2)
                                            if len(parts) == 3 and parts[0].strip().isdigit():
                                                current_linenumber = int(parts[0].strip())
                                                if int(parts[0].strip()) > linenumber_temp:
                                                    break
                                                if current_linenumber == int(linenumber_temp):
                                                    if lineindex_start == -1:
                                                        lineindex_start = index
                                                    lineindex_end = index
                                f_ast.close()

                                if lineindex_start == -1:
                                    continue
                                if lineindex_start == lineindex_end:
                                    list_ast = lines_ast[lineindex_start]
                                else:
                                    list_ast = lines_ast[lineindex_start: lineindex_end+1]

                                text_ast = ''.join(list_ast)
                                json_tree_ast = parse_text_to_json(list_ast)
                                target_name = "AssignmentExpression:"
                                json_tree_ast = json.loads(json_tree_ast)
                                sub_json_tree_AssignmentExpression = find_node(json_tree_ast, target_name) 
                                if sub_json_tree_AssignmentExpression is None:
                                    continue
                                if len(sub_json_tree_AssignmentExpression["children"]) != 2:
                                    continue
                                else:
                                    sub_json_tree_AssignmentExpression_left = sub_json_tree_AssignmentExpression["children"][0]
                                    sub_json_tree_AssignmentExpression_right = sub_json_tree_AssignmentExpression["children"][1]
                                    list_potential_variables_left = []
                                    names_list_left = get_names(sub_json_tree_AssignmentExpression_left)
                                    Callee_str = ""
                                    for line_ast_temp in names_list_left:
                                        if not line_ast_temp.startswith("None") and not line_ast_temp.startswith(":") and line_ast_temp.count(':') >= 2:
                                            parts = line_ast_temp.split(":", 2)
                                            if len(parts) == 3 and parts[0].strip().isdigit():
                                                if parts[1].strip() == "Callee":
                                                    Callee_str = str(parts[2].strip())
                                                if parts[1].strip() == "UnaryExpression" and parts[2].strip() != Callee_str and not str(parts[2].strip()).isupper() and linenumber_temp == int(parts[0].strip()):
                                                    list_potential_variables_left.append(str(parts[2].strip()))
                                                if parts[1].strip() == "Identifier" and parts[2].strip() != Callee_str and not str(parts[2].strip()).isupper() and linenumber_temp == int(parts[0].strip()):
                                                        list_potential_variables_left.append(str(parts[2].strip()))
                                                if parts[1].strip() == "ArrayIndexing" and parts[2].strip() != Callee_str and not str(parts[2].strip()).isupper() and linenumber_temp == int(parts[0].strip()):
                                                    list_potential_variables_left.append(str(parts[2].strip()))
                                                if parts[1].strip() == "MemberAccess" and parts[2].strip() != Callee_str and not str(parts[2].strip()).isupper() and linenumber_temp == int(parts[0].strip()):
                                                    list_potential_variables_left.append(str(parts[2].strip()))
                                            else:
                                                if line_ast_temp.startswith("None:Callee:"):
                                                    parts = line_ast_temp.split(":", 2)
                                                    if len(parts) == 3 and parts[1].strip() == "Callee" and parts[0].strip() == "None":
                                                        Callee_str = str(parts[2].strip())
                                                        continue
                                    list_potential_variables_left = remove_duplicates(list_potential_variables_left)
                                    if len(list_potential_variables_left) > 0:
                                        set3 = set(list_potential_variables_left)
                                        common_elements_left = set3.intersection(set2)
                                        if len(list(common_elements_left)) == 0:
                                            continue 
                                        else:
                                            if common_elements_temp in list(common_elements_left):
                                                flag_keyvariable_left_or_right = 1
                                    else:
                                        continue
                                    list_potential_variables_right = []
                                    names_list_right = get_names(sub_json_tree_AssignmentExpression_right)
                                    Callee_str = ""
                                    for line_ast_temp in names_list_right:
                                        if not line_ast_temp.startswith("None") and not line_ast_temp.startswith(":") and line_ast_temp.count(':') >= 2:
                                            parts = line_ast_temp.split(":", 2)
                                            if len(parts) == 3 and parts[0].strip().isdigit():
                                                if parts[1].strip() == "Callee":
                                                    Callee_str = str(parts[2].strip())
                                                if parts[1].strip() == "UnaryExpression" and parts[2].strip() != Callee_str and not str(parts[2].strip()).isupper() and linenumber_temp == int(parts[0].strip()):
                                                    list_potential_variables_right.append(str(parts[2].strip()))
                                                if parts[1].strip() == "Identifier" and parts[2].strip() != Callee_str and not str(parts[2].strip()).isupper() and linenumber_temp == int(parts[0].strip()):
                                                        list_potential_variables_right.append(str(parts[2].strip()))
                                                if parts[1].strip() == "ArrayIndexing" and parts[2].strip() != Callee_str and not str(parts[2].strip()).isupper() and linenumber_temp == int(parts[0].strip()):
                                                    list_potential_variables_right.append(str(parts[2].strip()))
                                                if parts[1].strip() == "MemberAccess" and parts[2].strip() != Callee_str and not str(parts[2].strip()).isupper() and linenumber_temp == int(parts[0].strip()):
                                                    list_potential_variables_right.append(str(parts[2].strip()))
                                            else:
                                                if line_ast_temp.startswith("None:Callee:"):
                                                    parts = line_ast_temp.split(":", 2)
                                                    if len(parts) == 3 and parts[1].strip() == "Callee" and parts[0].strip() == "None":
                                                        Callee_str = str(parts[2].strip())
                                                        continue
                                    list_potential_variables_right = remove_duplicates(list_potential_variables_right)
                                    if len(list_potential_variables_right) > 0:
                                        set4 = set(list_potential_variables_right)
                                        common_elements_right = set4.intersection(set2)
                                        if len(list(common_elements_right)) == 0:
                                            continue 
                                        else:
                                            if common_elements_temp in list(common_elements_right):
                                                flag_keyvariable_left_or_right = 2
                                                list_common_elements_right_new = []
                                                for temp in list(common_elements_right):
                                                    if common_elements_temp in temp or temp in common_elements_temp: 
                                                        continue
                                                    else:
                                                        list_common_elements_right_new.append(temp)
                                                if len(list_common_elements_right_new) > 0:
                                                    continue
                                                else: 
                                                    for common_elements_left_temp in list(common_elements_left):
                                                        subGraph_withlinenumber_left = next((dct[str(common_elements_left_temp)] for dct in list_subGraph_withLineNumber if str(common_elements_left_temp) in dct), None)
                                                        if subGraph_withlinenumber_left is None:
                                                            continue
                                                        else:
                                                            list_line_temp_new = []
                                                            for line_temp in subGraph_withlinenumber_left: 
                                                                if int(line_temp) >= int(linenumber_temp): 
                                                                    list_line_temp_new = list_line_temp_new + dict_ast_analyse[str(line_temp).strip()]
                                                            list_line_temp_new = [str(item) for item in list_line_temp_new]
                                                            subGraph_withlinenumber_original = subGraph_withlinenumber
                                                            subGraph_withlinenumber = subGraph_withlinenumber + list_line_temp_new
                                                            subGraph_withlinenumber = remove_duplicates(subGraph_withlinenumber)
                                                            subGraph_withlinenumber = sorted(subGraph_withlinenumber, key=int) 
                                                            break
                                            else:
                                                if flag_keyvariable_left_or_right == 1:
                                                    list_common_elements_right_new = []
                                                    list_common_elements_right_new = list(common_elements_right)
                                                    if len(list_common_elements_right_new) == 1:
                                                        subGraph_withlinenumber_right = next((dct[str(list_common_elements_right_new[0])] for dct in list_subGraph_withLineNumber if str(list_common_elements_right_new[0]) in dct), None)
                                                        if subGraph_withlinenumber_right is None:
                                                            continue
                                                        else:
                                                            list_line_temp_new = []
                                                            for line_temp in subGraph_withlinenumber_right: 
                                                                if int(line_temp) <= int(linenumber_temp): 
                                                                    list_line_temp_new = list_line_temp_new + dict_ast_analyse[str(line_temp).strip()]
                                                            list_line_temp_new = [str(item) for item in list_line_temp_new]
                                                            subGraph_withlinenumber_original = subGraph_withlinenumber
                                                            subGraph_withlinenumber = subGraph_withlinenumber + list_line_temp_new
                                                            subGraph_withlinenumber = remove_duplicates(subGraph_withlinenumber)
                                                            subGraph_withlinenumber = sorted(subGraph_withlinenumber, key=int) 
                                                    else:
                                                        continue
                                                else:
                                                    continue
                                    else:
                                        continue
                                
                      
                        if len(subGraph_withlinenumber) > 0:
                            if len(subGraph_withlinenumber_original) == 0:
                                if check_rootstatement_in_subgraph(subGraph_withlinenumber, statement_linenumber_start, statement_linenumber_stop): 
                                    list_subGraph_withlinenumber_in_statement.append(subGraph_withlinenumber)
                            else:
                                if subGraph_withlinenumber_original == subGraph_withlinenumber:
                                    if check_rootstatement_in_subgraph(subGraph_withlinenumber, statement_linenumber_start, statement_linenumber_stop): 
                                        list_subGraph_withlinenumber_in_statement.append(subGraph_withlinenumber)
                                else:
                                    if check_rootstatement_in_subgraph(subGraph_withlinenumber, statement_linenumber_start, statement_linenumber_stop): 
                                        list_subGraph_withlinenumber_in_statement.append(subGraph_withlinenumber)
                                    if check_rootstatement_in_subgraph(subGraph_withlinenumber_original, statement_linenumber_start, statement_linenumber_stop): 
                                        list_subGraph_withlinenumber_in_statement.append(subGraph_withlinenumber_original)
                
                list_subGraph_withlinenumber_in_statement = remove_duplicates(list_subGraph_withlinenumber_in_statement)
                if len(list_subGraph_withlinenumber_in_statement) > 0:
                    dict_temp = dict()
                    dict_temp["statement"] = statement
                    dict_temp["statement_linenumber"] = statement_linenumber
                    dict_temp["statement_similarity"] = statement_similarity
                    dict_temp["only_one_statement"] = only_one_statement
                    dict_temp["statement_linenumber_start"] = statement_linenumber_start
                    dict_temp["statement_linenumber_stop"] = statement_linenumber_stop
                    dict_temp["list_potential_variables_in_statement"] = list_potential_variables_in_statement
                    dict_temp["new_common_elements_list"] = new_common_elements_list
                    dict_temp["list_subGraph_withlinenumber_in_statement"] = list_subGraph_withlinenumber_in_statement
                    list_top_statements_new.append(dict_temp)
    
    return list_top_statements_new


def main(result_jsonfilepath, result_jsonfilepath_new, norm_mask_embedding_reshaped_1, model, threshold_getKeyVariable, json_path_all):

    with open(result_jsonfilepath, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    f.close()

    with open(json_path_all, 'r', encoding='utf-8') as f:
        json_data_all = json.load(f)
    f.close()

    target_functions_list = json_data["target_functions_list"]
    target_functions_list_new = []
    for index, dict_target_function in enumerate(target_functions_list):
        path_c = dict_target_function["target_function_filepath"]
        path_ast = path_c + ".ast"
        path_pdgdot = path_c + ".dot"
        len_edges = 1
        list_top_statements_new = generate_subGraph_data(path_pdgdot, path_ast, path_c, len_edges, dict_target_function["list_top_statements"], norm_mask_embedding_reshaped_1, model, threshold_getKeyVariable, json_data_all)
        
        if (len(list_top_statements_new) > 0): 
            dict_target_function["list_top_statements"] = list_top_statements_new
            target_functions_list_new.append(dict_target_function)
    json_data["target_functions_list"] = target_functions_list_new

    json_data_new = json.dumps(json_data, indent=4, ensure_ascii=False)
    with open(result_jsonfilepath_new, "w", encoding="utf-8") as f:
        f.write(json_data_new)
    f.close()
    
    return json_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='get_rootStatement_keyVariables')

    parser.add_argument('--threshold_getKeyVariable', type=float, default=0.5, help='threshold_getKeyVariable')
    parser.add_argument('--input_seed_with_mask', type=str, default="", help='input_seed_with_mask')
    parser.add_argument('--folder_json_path', type=str, default="../get_rootStatement_keyVariable/results/", help='folder_json_path')
    parser.add_argument('--input_seed_function_filename', type=str, default="", help='input_seed_function_filename')
    parser.add_argument('--folder_json_path_new', type=str, default="../get_slice/results/", help='folder_json_path_new')
    parser.add_argument('--model_path', type=str, default="", help='model_path')
    parser.add_argument('--json_path_all', type=str, default="../get_slice/results/get_slice-all.json", help='folder_json_path_all')

    args = parser.parse_args()

    threshold_getKeyVariable = args.threshold_getKeyVariable
    input_seed_with_mask = args.input_seed_with_mask
    folder_json_path = args.folder_json_path
    input_seed_function_filename = args.input_seed_function_filename
    folder_json_path_new = args.folder_json_path_new
    model_path = args.model_path
    json_path_all = args.json_path_all

    if ".c" in input_seed_function_filename:
        json_name = "get_rootStatement_keyVariable-" + input_seed_function_filename.strip()[:-2] + ".json" 
    else:
        json_name = "get_rootStatement_keyVariable-" + input_seed_function_filename.strip() + ".json" 
    result_jsonfilepath = os.path.join(folder_json_path, json_name)

    if ".c" in input_seed_function_filename:
        json_name_new = "get_slice-" + input_seed_function_filename.strip()[:-2] + ".json" 
    else:
        json_name_new = "get_slice-" + input_seed_function_filename.strip() + ".json" 
    result_jsonfilepath_new = os.path.join(folder_json_path_new, json_name_new)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniXcoder(model_path)
    model.to(device)

    with torch.no_grad():
        encoded_input_1 = model.tokenizer(input_seed_with_mask, return_tensors='pt')
        tokens_ids = model.tokenize([input_seed_with_mask],max_length=1023,mode="<encoder-only>")
        source_ids = torch.tensor(tokens_ids).to(device)
        tokens_embeddings,input_seed_embedding = model(source_ids)
        mask_token_id = model.tokenizer.convert_tokens_to_ids("<mask>")
        mask_token_position = encoded_input_1['input_ids'].tolist()[0].index(mask_token_id)
        mask_embedding = tokens_embeddings[0, mask_token_position, :]
        mask_embedding_reshaped = mask_embedding.unsqueeze(0)
        norm_mask_embedding_reshaped_1 = torch.nn.functional.normalize(mask_embedding_reshaped, p=2, dim=1)

        main(result_jsonfilepath, result_jsonfilepath_new, norm_mask_embedding_reshaped_1, model, threshold_getKeyVariable, json_path_all)

    endTime = datetime.datetime.now()
    print("start time: ", startTime)
    print("end time: ", endTime)
    diffrentTime = (endTime - startTime).seconds
    print("different time: ", diffrentTime, "s")






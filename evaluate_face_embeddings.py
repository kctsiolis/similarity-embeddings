#Evaluation of face embeddings on binary classification task 
#The task is to determine whether or not two faces are identical
#Assuming that face embeddings are originally stored as numpy array (not pickle)
#Based on code by Tiago Salvador

import numpy as np
import pandas
from sklearn.metrics import roc_auc_score
import argparse
from compare_kernels import load_matrix, l2_normalization

def get_args(parser):
    parser.add_argument('-o', dest='dataset', choices=['rfw', 'bfw'], help='Dataset to use (either rfw or bfw).')
    parser.add_argument('--data_path', dest='data_path', help='Path to directory containing dataset csv file.')
    parser.add_argument('--embs_path', dest='embs_path', help='Path to embeddings.')

    return parser

def runner(args):
    dataset = args.dataset
    data_path = args.data_path
    embs_path = args.embs_path

    embs = load_matrix(embs_path)

    if dataset == 'rfw':
        db_cal = pandas.read_csv(data_path + '/rfw.csv')
        print("Building dictionary...")
        rfw_dict = build_rfw_dict(db_cal)
        print("Dictionary built.")
        print("Evaluating...")
        auc = evaluate_rfw(rfw_dict, db_cal, embs)
    else:
        db_cal = pandas.read_csv(data_path + '/bfw.csv')
        print("Building dictionary...")
        bfw_dict = build_bfw_dict(db_cal)
        print("Dictionary built.")
        print("Evaluating...")
        auc = evaluate_bfw(bfw_dict, db_cal, embs)

    print("AUC: {}".format(auc))

#Build dictionary which associates ids with row indices in the embedding matrix
def build_rfw_dict(db_cal):
    counter = 0
    rfw_dict = {}
    faces_id_num = []
    for subgroup in ['African', 'Asian', 'Caucasian', 'Indian']:
        select = db_cal['ethnicity'] == subgroup
        for id_face, num_face in zip(['id1', 'id2'], ['num1', 'num2']):
            folder_names = db_cal[select][id_face].values
            file_names = db_cal[select][id_face] + '_000' + db_cal[select][num_face].astype(str) + '.jpg'
            file_names = file_names.values
            for folder_name, file_name in zip(folder_names, file_names):
                if file_name not in faces_id_num:
                    rfw_dict[(id_face, num_face)] = counter
                    counter += 1
                    faces_id_num.append(file_name)

    return rfw_dict

#Build dictionary which associates paths with row indices in the embedding matrix
def build_bfw_dict(db_cal):
    counter = 0
    bfw_dict = {}
    file_names_visited = []
    for path in ['path1', 'path2']:
        file_names = db_cal[path].values
        for file_name in file_names:
            if file_name not in file_names_visited:
                bfw_dict[file_name] = counter
                counter += 1
                file_names_visited.append(file_name)

    return bfw_dict

def evaluate_rfw(dict, db_cal, embs):
    y_true = []
    y_score = []

    for id1, id2, num1, num2, label in zip(db_cal['id1'].values, db_cal['id2'].values,
        db_cal['num1'].values, db_cal['num2'].values, db_cal['pair'].values):
        idx1 = dict[(id1, num1)]
        idx2 = dict[(id2, num2)]

        if label == 'Genuine':
            y_true.append(1)
        else:
            y_true.append(0)

        score = np.dot(embs[idx1,:], embs[idx2,:])
        y_score.append(score)

    return roc_auc_score(y_true, y_score)  

def evaluate_bfw(dict, db_cal, embs):
    y_true = []
    y_score = []

    for p1, p2, label in zip(db_cal['path1'].values, db_cal['path2'].values, db_cal['pair'].values):
        idx1 = dict[p1]
        idx2 = dict[p2]
        
        if label == 'Genuine':
            y_true.append(1)
        else:
            y_true.append(0)

        score = np.dot(embs[idx1,:], embs[idx2,:])
        y_score.append(score)

    return roc_auc_score(y_true, y_score)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = get_args(parser)
    args = parser.parse_args()
    runner(args)
    
#Modified from the original code provided by Tiago Salvador, used with his permission

import pandas
import pickle
import numpy as np

#Functions to load embeddings

def collect_embeddings_rfw(feature, db_cal, embs_path):
    # collect embeddings of all the images in the calibration set
    embeddings = np.zeros((0, 512))  # all embeddings are in a 512-dimensional space
    faces_id_num = []
    if feature != 'arcface':
        for subgroup in ['African', 'Asian', 'Caucasian', 'Indian']:
            temp = pickle.load(open(embs_path + 'rfw/' + subgroup + '_' + feature + '_embeddings.pickle', 'rb'))
            select = db_cal['ethnicity'] == subgroup
            for id_face, num_face in zip(['id1', 'id2'], ['num1', 'num2']):
                folder_names = db_cal[select][id_face].values
                file_names = db_cal[select][id_face] + '_000' + db_cal[select][num_face].astype(str) + '.jpg'
                file_names = file_names.values
                for folder_name, file_name in zip(folder_names, file_names):
                    key = 'rfw/data/' + subgroup + '_cropped/' + folder_name + '/' + file_name
                    if file_name not in faces_id_num:
                        embeddings = np.concatenate((embeddings, temp[key]))
                        faces_id_num.append(file_name)
    else:
        temp = pickle.load(open(embs_path + 'rfw/rfw_' + feature + '_embeddings.pickle', 'rb'))
        for subgroup in ['African', 'Asian', 'Caucasian', 'Indian']:
            select = db_cal['ethnicity'] == subgroup
            for id_face, num_face in zip(['id1', 'id2'], ['num1', 'num2']):
                folder_names = db_cal[select][id_face].values
                file_names = db_cal[select][id_face] + '_000' + db_cal[select][num_face].astype(str) + '.jpg'
                file_names = file_names.values
                for folder_name, file_name in zip(folder_names, file_names):
                    key = 'rfw/data/' + subgroup + '/' + folder_name + '/' + file_name
                    if file_name not in faces_id_num:
                        embeddings = np.concatenate((embeddings, temp[key].reshape(1, -1)))
                        faces_id_num.append(file_name)

    return embeddings

def collect_embeddings_bfw(feature, db_cal, embs_path):
    # collect embeddings of all the images in the calibration set
    embeddings = np.zeros((0, 512))  # all embeddings are in a 512-dimensional space
    file_names_visited = []
    temp = pickle.load(open(embs_path + 'bfw/' + feature + '_embeddings.pickle', 'rb'))
    for path in ['path1', 'path2']:
        file_names = db_cal[path].values
        for file_name in file_names:
            if file_name not in file_names_visited:
                embeddings = np.concatenate((embeddings, temp[file_name].reshape(1, -1)))
                file_names_visited.append(file_name)

    return embeddings

#Functions to save the embeddings

if __name__ == '__main__':
    load_path = '/mnt/data/scratch/konstantinos.tsiolis/face_embeddings/Embeddings/'
    save_path = '/mnt/data/scratch/konstantinos.tsiolis/face_embeddings/arcface_bfw.npy'
    #emb_types = ['facenet', 'facenet-webface', 'arcface']

    '''
    #Load csv file into pandas dataframe
    print("Loading csv...")
    rfw_df = pandas.read_csv(load_path + 'rfw/rfw.csv')
    print("Loaded csv.")

    print("Collecting embeddings...")
    rfw_embs = collect_embeddings_rfw('arcface', rfw_df, load_path)
    print("Embeddings collected.")

    np.save(save_path, rfw_embs)
    '''

    #Load csv file into pandas dataframe
    print("Loading csv...")
    bfw_df = pandas.read_csv(load_path + '/bfw/bfw.csv')  
    print("Loaded csv.")

    print("Collecting embeddings...")
    bfw_embs = collect_embeddings_bfw('arcface', bfw_df, load_path)
    print("Embeddings collected.")

    np.save(save_path, bfw_embs)
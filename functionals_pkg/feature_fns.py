import numpy as np
from tqdm import tqdm

def lexsort_based_unique(data):
    sorted_data =  data[np.lexsort(data.T),:]
    row_mask = np.append([True],np.any(np.diff(sorted_data,axis=0),1))
    return sorted_data[row_mask]

def make_dictionary(list_cuboids, kmeans_obj, model):

    print "$$$$$$$$$$$$$$$$$$$$$$$$"
    print "GENERATING DICTIONARY"
    print "$$$$$$$$$$$$$$$$$$$$$$$$"

    word_list = []
    for i in tqdm(range(0, len(list_cuboids))):

        rows = list_cuboids[i][0].shape[0]
        cols = list_cuboids[i][0].shape[1]

        for l in range(1, len(list_cuboids[i]) - 1):
            for j in range(1, rows - 1):
                for k in range(1, cols - 1):

                    surroundings = []

                    surr_idx = l - 1
                    current_cuboid = list_cuboids[i][l][j, k]

                    surroundings.append(current_cuboid)

                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k - 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k + 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k - 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k + 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k - 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k + 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k])

                    surr_idx = l
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k - 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k + 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k - 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k + 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k - 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k + 1])


                    surr_idx = l + 1
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k - 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k + 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k - 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k + 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k - 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k + 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k])

                    data_array = np.array(surroundings)
                    surrounding_feats = model.predict(data_array)
                    word = kmeans_obj.predict(surrounding_feats)

                    word_list.append(word.tolist())



    return word_list

def dictionary_based_anom_setting(list_cuboids,model,kmeans_obj,dictionary,mean_data, std_data,mean_feats,std_feats):

    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
    print "GENERATING DICTIONARY BASED ANOMALY SETTINGS"
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

    list_of_frequencies = [0]*len(dictionary)

    for i in tqdm(range(0, len(list_cuboids))):

        rows = list_cuboids[i][0].shape[0]
        cols = list_cuboids[i][0].shape[1]

        for l in range(1, len(list_cuboids[i]) - 1):
            for j in range(1, rows - 1):
                for k in range(1, cols - 1):

                    surroundings = []

                    surr_idx = l - 1
                    current_cuboid = list_cuboids[i][l][j, k]

                    surroundings.append(current_cuboid)

                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k - 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k + 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k - 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k + 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k - 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k + 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k])

                    surr_idx = l
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k - 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k + 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k - 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k + 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k - 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k + 1])


                    surr_idx = l + 1
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k - 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j - 1, k + 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k - 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k + 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k - 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j + 1, k + 1])
                    surroundings.append(
                        list_cuboids[i][surr_idx][j, k])

                    data_array = np.array(surroundings)
                    surrounding_feats = model.predict(data_array)
                    word = kmeans_obj.predict(surrounding_feats)
                    word = word.tolist()

                    if(word in dictionary):
                        idx = dictionary.index(word)
                        # print "word:",word, "matches with ", dictionary[idx]
                        list_of_frequencies[idx]=list_of_frequencies[idx]+1
                    else:
                        # print "word:",word, " does not match with any dictionary entry."
                        list_cuboids[i][l][j, k].anom_score = 2.0

                    list_cuboids[i][l][j, k].update_status()


    return list_cuboids, list_of_frequencies


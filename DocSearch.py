import numpy as np
import math
import time


def main():
    """
    This is the main method which calls upon all the other methods of this module to accomplish tasks.
    """
    start_time = time.time()

    docs_file = read_txt_file("docs.txt")
    query_list = read_txt_file("queries.txt")

    # print(query_list)

    # 1) create an inverted index for the document corpus
    global_inverted_index = create_inverted_index(docs_file)
    print("Words in the dictionary:\t", len(global_inverted_index))
    print()

    for query in query_list:
        print("Query:\t", query)
        # 2) find relevant document IDs for the query
        set_of_relevant_document_ids = find_relevant_document(global_inverted_index, query)

        print("Relevant Documents:\t", end="")
        [print((x+1), end=" ") for x in set_of_relevant_document_ids]
        print()

        if set_of_relevant_document_ids:

            # 3) from the docs_file, using the relevant document IDs, get each document and
            # append it to the dictionary, relevant_documents. Where key=docID and value=[a,b,...z].
            relevant_documents_dict = dict()
            for doc_id in set_of_relevant_document_ids:
                relevant_documents_dict[doc_id] = docs_file[doc_id]

            # print(relevant_documents_dict)
            # 4) create a word count dictionary for each document and also a common dict mask
            returned_tuple = create_dictionary(relevant_documents_dict)
            dict_of_relevant_docs = returned_tuple[0]

            common_dictionary_mask = returned_tuple[1]

            # 5) Rank the documents with the query from most to least relevance
            rank_documents(dict_of_relevant_docs, query, common_dictionary_mask)

        else:
            print("No relevant documents IDs found!")

        print("\n")

    print("--- %s seconds ---" % (time.time() - start_time))


def read_txt_file(filename):
    """
    In summary, this func reads a txt file and strips the newline character and returns a list of each line.
    Each line of the docs.txt is a document. The readlines() reads each line aka document and appends it to a list.
    The data variable holds a list -> you can verify using the >>> type(data). Each list element represent a document.
    The range() function does not an explicit start value is not specified so python assumes it to be 0. len(data)
    specifies the max range for range(). The returned list data contains sanitized data.
    """

    with open(filename, "r") as f:
        data = f.readlines()

    print(data)

    for index in range(len(data)):
        temp_line = data[index]
        data[index] = temp_line.strip('\n')
    return data


def create_inverted_index(document_list):
    """
    takes in a list containing documents and returns an inverted index
    which is basically a dictionary, where key = term, value = list[document_id's]
    {'word':[1,2,5], '...':[...], ...}
    """

    inverted_index = dict()

    for a in range(len(document_list)):

        temp_document = document_list[a].split()
        # print("splitted docs : ", temp_document)

        # if its the first document add every word straight to the inv index (dictionary) with value = 0.
        if a == 0:
            # dictionary comprehension
            inverted_index = {key: list([0]) for key in temp_document}

        # else loop through the dict...
        else:
            for b in range(len(temp_document)):

                # ...if the key exists in the inverted index, then if the document ID is not already present,
                # append the document ID
                # if the key doesnt exist, create it and assign the document ID contained in the list
                if inverted_index.get(temp_document[b]):
                    if not(a in inverted_index[temp_document[b]]):
                        inverted_index[temp_document[b]].append(a)
                else:
                    inverted_index[temp_document[b]] = [a]

    return inverted_index


def print_inverted_index(inverted_index):
    """prints the inverted index"""
    print("Inverted index: ", end="")
    for i in inverted_index:
        print("\t"*4, i, "Doc ID's:", inverted_index[i], sep=" | ")


def find_relevant_document(inverted_index, query):
    """
    In summary, this func returns a set of relevant documents for all the words in the query.
    Now we have a set of documents that are relevant to the query terms.
    """

    not_found_flag = False
    query_set = set(query.split())

    list_of_document_ids = list()
    for term in query_set:
        if inverted_index.get(term):
            list_of_document_ids.append(set(inverted_index[term].copy()))
        else:
            not_found_flag = True
            break

    if not(not_found_flag):
        # take the intersection of the sets from the list_of_document_ids

        set_of_doc_ids = list_of_document_ids[0]

        # If there's more than 1 item in the list take the intersection of all of them
        # so that ONLY the document(s) which contain all the query terms are returned.
        if len(list_of_document_ids) > 1:
            temp_set = list_of_document_ids[0]
            for x in range(1, len(list_of_document_ids)):
                temp_set = temp_set & list_of_document_ids[x]

            set_of_doc_ids = temp_set

        return set_of_doc_ids

    return


def create_dictionary(dict_of_docs):
    """
    creates a dictionary, docDict, for each relevant document and appends docDict to a bigger
    dictionary, where key = docID and value= docDict. The docDict, which is the inner dictionary, contains
    the word count for a given document, where key = word and value = count.
    { docID:{'word':count, 'word':count}, docID2:{...}, ...}
    secondly it also creates the common dictionary mask.
    Returns a tuple.
    """

    set_of_words = set()
    outer_dict = dict()
    for docID in dict_of_docs:
        temp_line = dict_of_docs[docID]
        temp_line = temp_line.split()

        inner_dict = dict()

        for word in temp_line:
            if inner_dict.get(word):
                inner_dict[word] += 1
            else:
                inner_dict[word] = 1

            set_of_words.add(word)

        # append document to the outer dictionary using docID as the dict key = document ID
        outer_dict[docID] = inner_dict

    common_dictionary = dict().fromkeys(set_of_words, 0)

    return outer_dict, common_dictionary


def create_common_dictionary(dictionary_mask, document_dict):
    """
    creates a word count dictionary, for a document, where all the keys are based on
    all the relevant documents' words
    """
    temp_common_dict = dictionary_mask.copy()
    for key in document_dict:
        if document_dict[key] != 0:
            temp_common_dict[key] = document_dict[key]

    return temp_common_dict


def rank_documents(dict_of_relevant_docs, query, common_dictionary_mask):
    """
    Based off the common dictionary mask(derived from the set of relevant documents):
        First it creates a word count dictionary for the search query using the common
        Next for each relevant document, create a common word count dictionary.
            Then workout the angle of the query and the document.
        Next sort the list from asc order for each document by the angle.
        Print the sorted list.
    """

    # create a word count dictionary for the query using the common dict mask.
    # The query is treated as a document. **assuming every word only appears once in the query**
    query_dictionary = dict().fromkeys(query.split(), 1)
    query_common_dictionary = create_common_dictionary(common_dictionary_mask, query_dictionary)
    nparray_query = np.array(list(query_common_dictionary.values()))
    # print("Common QUERY dict: ", query_common_dictionary)
    # print("Common QUERY Vectors: ", np.array(list(query_common_dictionary.values())))

    # **assuming every word only appears once in the query**
    query_dot = np.array([1 for n in range(len(query.split()))])

    list_of_angles = list()
    for doc_id in dict_of_relevant_docs:
        temp_doc_common_dictionary = create_common_dictionary(common_dictionary_mask, dict_of_relevant_docs[doc_id])
        # print("Common DOC Dict: ", temp_doc_common_dictionary)
        # print("Common DOC Vectors: ", np.array(list(temp_doc_common_dictionary.values())))

        query_list = query.split()
        doc_dot = np.array([temp_doc_common_dictionary[a_query] for a_query in query_list])

        angle = calc_angle(query_dot, doc_dot, nparray_query, np.array(list(temp_doc_common_dictionary.values())))
        list_of_angles.append([doc_id, angle])

    sorted_list_of_angles = sorted(list_of_angles, key=lambda item: item[1])

    for x in sorted_list_of_angles:
        print("{0} \t {1:.2f} ".format((x[0] +1), x[1]))


    return


def calc_angle(dot_query, dot_doc, search_query_vector, document_vector):
    """
    *** Vector based document ranking ***
    The function calculates the ranking (angle) of a document with the search query.
    Theta represents the angle between the 2 vectors. The smaller the angle the more
    relevant the document.
    """
    # Aliases
    x = search_query_vector
    y = document_vector

    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    cos_theta = np.dot(dot_query, dot_doc) / (norm_x * norm_y)
    theta = math.degrees(math.acos(cos_theta))
    return round(theta, 2)

main()

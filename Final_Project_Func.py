#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy  as np
import re 
from google.cloud import vision

# In[10]:


def detectText(img):
    with io.open(img,'rb') as image_file:
        content = image_file.read()

    image=vision.types.Image(content=content)
    response = client.text_detection(image=image) # returns TextAnnotation
    texts = response.text_annotations
    
    # annotate Image Response
    df = pd.DataFrame(columns=['locale', 'description'])
    for text in texts:
        df = df.append(
            dict(
                locale=text.locale,
                description=text.description
            ),
            ignore_index=True
    )

    return df 


# In[13]:


def image_by_rows(response):
#reads the image by rows, using x & y coordinates     
    items = []
    lines = {}

    for text in response.text_annotations[1:]:
        top_x_axis = text.bounding_poly.vertices[0].x
        top_y_axis = text.bounding_poly.vertices[0].y
        bottom_y_axis = text.bounding_poly.vertices[3].y

        if top_y_axis not in lines:
            lines[top_y_axis] = [(top_y_axis, bottom_y_axis), []]

        for s_top_y_axis, s_item in lines.items():
            if top_y_axis < s_item[0][1]:
                lines[s_top_y_axis][1].append((top_x_axis, text.description))
                break

    for _, item in lines.items():
        if item[1]:
            words = sorted(item[1], key=lambda t: t[0])
            items.append((item[0], ' '.join([word for _, word in words]), words))
            
    df = pd.DataFrame.from_records(items)
    return df


# In[2]:


#Separate the value and its label from the string
def get_label_from_string(string):
    label_arr = re.findall("([A-Z][a-zA-Z]*)", string)
    label_name = ""
    label_value = ""

    if len(label_arr) == 0:
        label_name = "|"+string+'|'
    elif len(label_arr) == 1:
        label_name = label_arr[0]
    else:
        label_name = label_arr[0] + ' ' + label_arr[1]

    digit_pattern = "[-+]?\d*\.,\d+g|\d+,"
    value_arr = re.findall("{0}g|{0}%|{0}J|{0}kJ|{0}mg|{0}kcal".format(digit_pattern), string)
    # print(value_arr)
    if len(value_arr):
        label_value = value_arr[0]
    else:
        label_value = "|"+string+'|'
    return label_name, label_value


# In[3]:


def create_dict(df):
    dict1 = {}
    for row in df[1]:
        tuple_ = get_label_from_string(row)
        dict1[tuple_[0]] = tuple_[1]
    return dict1


# In[4]:


#Removes all the unnecessary noise from a string
def clean_string(string):
    pattern = "[\|\*\_\'\—\-\{}.,/]".format('"')
    text = re.sub(pattern, " ", string)
    text = re.sub(" I ", " / ", text)
    text = re.sub("^I ", "", text)
    text = re.sub("Omg", "0mg", text)
    text = re.sub("Og", "0g", text)
    text = re.sub('(?<=\d) (?=\w)', '', text)
    text = text.strip()
    return text


# In[5]:


#Separate the unit from its value. (eg. '24g' to '24' and 'g')
def separate_unit(string):
    r1 = re.compile("(\d+[\.\,\']?\d*)([a-zA-Z]+)")
    m1 = r1.match(string)
    r2 = re.compile("(\d+[\.\,\']?\d*)")
    m2 = r2.match(string)
    if m1:
        return (float(m1.group(1).replace(',','.').replace("'",'.')), m1.group(2))
    elif m2:
        return (float(m2.group(1).replace(',','.').replace("'",'.')), str(''))
    else:
        return ('')


# In[6]:


# converting the tuple to the list
def tup_to_list(dict_):
    dict_list = {k: list(v) for k, v in dict_.items()}
    return dict_list


# In[8]:


# eliminate the second value from the list 
def clean_values(list_):
    if (len(list_) >= 2):
           list_.pop(1)
    return list_ 

def clean_dict(dict_list):
    # delete second element 
    _dict = {k: clean_values(v) for k, v in dict_list.items()} # 
    return _dict


# In[9]:


def make_dataframe(clean_dict):
    df = pd.DataFrame.from_dict(clean_dict, orient='index')
    df = df.transpose()
    return df 


# In[10]:

def remove_chars(s):
    return re.sub('[^0-9,]+', ' ', s) 

def clear_df(df):
    df_clean = df.apply(np.vectorize(remove_chars))
    return df_clean 


from nltk import word_tokenize
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('spanish')

def stem_sentences(sentence):
    tokens = sentence.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)



def rename_columns(df):

    df.columns = df.columns.to_series().apply(lambda x: 'energy-kcal_100g' if 'ener' in x else x)
    df.columns = df.columns.to_series().apply(lambda x: 'proteins_100g' if 'prot' in x else x)
    df.columns = df.columns.to_series().apply(lambda x: 'carbohydrates_100g' if 'hidrat' in x else x)
    df.columns = df.columns.to_series().apply(lambda x: 'salt_100g' if 'sal' in x else x)
    df.columns = df.columns.to_series().apply(lambda x: 'sugars_100g' if 'azucar' in x else x)
    df.columns = df.columns.to_series().apply(lambda x: 'mono' if 'monoinsat' in x else x)
    df.columns = df.columns.to_series().apply(lambda x: 'poli' if 'poli' in x else x)
    df.columns = df.columns.to_series().apply(lambda x: 'other' if 'dos quais satur' in x else x)
    df.columns = df.columns.to_series().apply(lambda x: 'other2' if 'saturi' in x else x)
    df.columns = df.columns.to_series().apply(lambda x: 'saturated-fat_100g' if 'satur' in x else x)
    df.columns = df.columns.to_series().apply(lambda x: 'fat_100g' if 'gras' in x else x)
    
    return df 


def image_to_df(img_url):
    
    client = vision.ImageAnnotatorClient()
    image = vision.types.Image()
    image.source.image_uri = img_url
    response = client.text_detection(image=image) # returns TextAnnotation

    df = image_by_rows(response) #returns a dataframe

    df = df.drop([0,2], axis=1)

    dict2 = create_dict(df)

    df_test = make_dataframe(dict2) # creates new dataframe

    df_clean = clear_df(df_test)
    
    df_clean.columns = df_clean.columns.to_series().apply(clean_string)
    
    df_clean.columns = df_clean.columns.to_series().apply(stem_sentences)
    
    df_clean = rename_columns(df_clean)

    return df_clean


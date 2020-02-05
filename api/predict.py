#
# Copyright 2018-2019 IBM Corp. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import sys
sys.path.append('/home/ihong/evalmaxner')
from core.model import ModelWrapper

from maxfw.core import MAX_API, PredictAPI, MetadataAPI
from flask_restplus import fields
from flask import request
import json
import csv
import re

model_wrapper = ModelWrapper()

# === Labels API

model_label = MAX_API.model('ModelLabel', {
    'id': fields.String(required=True, description='Label identifier'),
    'name': fields.String(required=True, description='Entity label'),
    'description': fields.String(required=False, description='Meaning of entity label')
})

labels_response = MAX_API.model('LabelsResponse', {
    'count': fields.Integer(required=True, description='Number of labels returned'),
    'labels': fields.List(fields.Nested(model_label), description='Entity labels that can be predicted by the model')
})

# Reference: http://gmb.let.rug.nl/manual.php
tag_desc = {
    'B-PER': 'Person; entities are limited to individuals that are human or have human characteristics, such as divine entities. B- tag indicates start of a new phrase.',  # noqa
    'I-PER': 'Person; entities are limited to individuals that are human or have human characteristics, such as divine entities.',  # noqa
    'B-GEO': 'Location; entities are limited to geographical entities such as geographical areas and landmasses, bodies of water, and geological formations. B- tag indicates start of a new phrase.',  # noqa
    'I-GEO': 'Location; entities are limited to geographical entities such as geographical areas and landmasses, bodies of water, and geological formations.',  # noqa
    'B-LOC': 'Location; entities are limited to geographical entities such as geographical areas and landmasses, bodies of water, and geological formations. B- tag indicates start of a new phrase.',  # noqa
    'I-LOC': 'Location; entities are limited to geographical entities such as geographical areas and landmasses, bodies of water, and geological formations.',  # noqa
    'B-ORG': 'Organization; entities are limited to corporations, agencies, and other groups of people defined by an established organizational structure. B- tag indicates start of a new phrase.',  # noqa
    'I-ORG': 'Organization; entities are limited to corporations, agencies, and other groups of people defined by an established organizational structure',  # noqa
    'B-GPE': 'Geo-political Entity; entities are geographical regions defined by political and/or social groups. A GPE entity subsumes and does not distinguish between a city, a nation, its region, its government, or its people. B- tag indicates start of a new phrase.',  # noqa
    'I-GPE': 'Geo-political Entity; entities are geographical regions defined by political and/or social groups. A GPE entity subsumes and does not distinguish between a city, a nation, its region, its government, or its people',  # noqa
    'B-TIM': 'Time; limited to references to certain temporal entities that have a name, such as the days of the week and months of a year. B- tag indicates start of a new phrase.',  # noqa
    'I-TIM': 'Time; limited to references to certain temporal entities that have a name, such as the days of the week and months of a year.',  # noqa
    'B-EVE': 'Event; incidents and occasions that occur during a particular time. B- tag indicates start of a new phrase.',  # noqa
    'I-EVE': 'Event; incidents and occasions that occur during a particular time.',
    'B-ART': 'Artifact; limited to manmade objects, structures and abstract entities, including buildings, facilities, art and scientific theories. B- tag indicates start of a new phrase.',  # noqa
    'I-ART': 'Artifact; limited to manmade objects, structures and abstract entities, including buildings, facilities, art and scientific theories.',  # noqa
    'B-NAT': 'Natural Object; entities that occur naturally and are not manmade, such as diseases, biological entities and other living things. B- tag indicates start of a new phrase.',  # noqa
    'I-NAT': 'Natural Object; entities that occur naturally and are not manmade, such as diseases, biological entities and other living things.',  # noqa
    'O': 'No entity type'
}


class ModelLabelsAPI(MetadataAPI):
    '''API for getting information about available entity tags'''
    @MAX_API.doc('get_labels')
    @MAX_API.marshal_with(labels_response)
    def get(self):
        '''Return the list of labels that can be predicted by the model'''
        result = {}
        result['labels'] = [{'id': l[0], 'name': l[1], 'description': tag_desc[l[1]] if l[1] in tag_desc else ''}
                            for l in model_wrapper.id_to_tag.items()]
        result['count'] = len(model_wrapper.id_to_tag)
        return result

# === Predict API


input_example = ['John lives SF here.', 'I ate apple.', 'I am a dancer and singer.', 'Model Asset Exchange NER model is popular than other models.', 'I like banana.', 'I ate a lot of apples.']
ent_example = ['I-PER', 'O', 'O', 'I-LOC', 'O', 'O', 'O', 'O', 'I-ORG']
term_example = ['John', 'lives', 'in', 'Brussels', 'and', 'works', 'for', 'the', 'EU']

# model_input = MAX_API.model('ModelInput', {
#     'text': fields.String(required=True, description='Text for which to predict entities', example=input_example)
# })

model_input = MAX_API.model('ModelInput', {
    'text': fields.List(fields.String, required=True,
                        description='Text for which to predict entities', example=input_example)
})

model_prediction = MAX_API.model('ModelPrediction', {
    'tags': fields.List(fields.String, required=True, description='List of predicted entity tags, one per term in the input text.',  # noqa
                        example=ent_example),
    'terms': fields.List(fields.String, required=True,
                         description='Terms extracted from input text pre-processing. Each term has a corresponding predicted entity tag in the "tags" field.',  # noqa
                         example=term_example)
})

predict_response = MAX_API.model('ModelPredictResponse', {
    'status': fields.String(required=True, description='Response status message'),
    'prediction': fields.Nested(model_prediction, description='Model prediction')
})


# class ModelPredictAPI(PredictAPI):
#
#     @MAX_API.doc('predict')
#     @MAX_API.expect(model_input)
#     @MAX_API.marshal_with(predict_response)
#     def post(self):
#         '''Make a prediction given input data'''
#         result = {'status': 'error'}
#
#         inp_text = [
#             "John lives SF here.",
#             "I ate apple.",
#             "I am a dancer and singer.",
#             "Model Asset Exchange NER model is popular than other models."
#             ]
#         #j = request.get_json()
#         text = inp_text
#         print(text)
#         entities, terms = model_wrapper.predict(text)
#         model_pred = {
#             'tags': entities,
#             'terms': terms
#         }
#         result['prediction'] = model_pred
#         result['status'] = 'ok'
#
#         return result


# with open('/Users/ihjhuo@ibm.com/nerbatch/MAX-Named-Entity-Tagger/en-50k-200.json', 'r') as myfile:
#     data = myfile.read()
# watson_test_data_obj = json.loads(data)

# watson_test_data_obj = [] 
# for line in open('en-50k-200.json', 'r'):
#    watson_test_data_obj.append(json.loads(line))   

# input_sentences = []
# for i in range(len(watson_test_data_obj)):
#    input_sentences.append(watson_test_data_obj[i]['text'])

### all doc sentences of en50-200
watson_test_data_obj = []

with open('en-50k-200.json_tokens.csv') as csvfile:    
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        lgth_row = len (row)
        for row_idx in range(lgth_row):
            if len(row[row_idx]) > 0 :
                # print('row single:', row[row_idx])
                testing_sentences = ""
                # words = row[row_idx].strip('][').split(', ') 

                words = [f[1:-1] for f in re.findall("'.+?'", row[row_idx])]

                if len(words) == 0:
                    continue
                # print('word', words)
                # print('word', type(words))
                # print('word', len(words))
                # print('word', words[0])
                
                for spt_sent in words:
                    testing_sentences = testing_sentences + spt_sent + " "

                # print('sentence:', testing_sentences)
                # print('sentence:', len(testing_sentences))
                watson_test_data_obj.append(testing_sentences)

        # print('a:',watson_test_data_obj)
        # print('a:',watson_test_data_obj[0])
        # print('a:',watson_test_data_obj[1])
        # print('a:',watson_test_data_obj[-2])
        # print('a:',watson_test_data_obj[-1])
        # sys.exit()

############################################################
# for line in open('en-50k-200.json', 'r'):
#     line_data = json.loads(line)
#     split_sentences = line_data['text'].split('\n')

#     print('HRE IS SPLIT SENTENCE: ', type(split_sentences))
#     print('HRE IS SPLIT SENTENCE: ', len(split_sentences))
#     print('HRE IS SPLIT SENTENCE: ', split_sentences[0])
#     print('HRE IS SPLIT SENTENCE: ', split_sentences[1])
#     print('HRE IS SPLIT SENTENCE: ', type(split_sentences[1]))

#     testing_sentences = []
#     for spt_sent in split_sentences:
#         if len(spt_sent) <= 3:
#             continue
#         else:
#             print("every word:", spt_sent)
#             testing_sentences.append(spt_sent)
#         print('here is: ', testing_sentences)

#     watson_test_data_obj.extend(testing_sentences)
#     # print('watson test: ', type(watson_test_data_obj))
#     # print('watson test: ', len(watson_test_data_obj))
#     # print('watson test: ', type(watson_test_data_obj[-1]))
#     print('watson test: ', watson_test_data_obj[0])
#     print('watson test: ', watson_test_data_obj[1])
#     print('watson test: ', watson_test_data_obj[-2])
#     print('watson test: ', watson_test_data_obj[-1])
#     sys.exit()
#################################################################

# inp_text = [
#             "John lives SF here.",
#             "I ate apple.",
#             "I am a dancer and singer.",
#             "Model Asset Exchange NER model is popular than other models.",
#             "I like banana.",
#             "I ate a lot of apples.",
            # ]

# text = inp_text
# print(text)

total_char = 0
for c_char in range(len(watson_test_data_obj)):
    number_tmp = len(watson_test_data_obj[c_char])
    total_char += number_tmp
print('TOTAL CHARACHERS:', total_char)
entities, terms, total_inftime = model_wrapper.predict(watson_test_data_obj)
model_pred = {
            'tags': entities,
            'terms': terms,
            'total_inftime':total_inftime
        }
print('throughput:',total_char/total_inftime)

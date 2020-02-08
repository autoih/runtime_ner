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

from maxfw.model import MAXModelWrapper

import numpy as np
import re
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

import logging
from core.utils import get_processing_word, load_vocab, pad_sequences
from config import DEFAULT_MODEL_PATH, MODEL_META_DATA as model_meta
import timeit
import pandas as pd
import mem_util
from tensorflow.python.client import timeline
import sys

logger = logging.getLogger()


class ModelWrapper(MAXModelWrapper):

    MODEL_META_DATA = model_meta

    pat = re.compile(r'(\W+)')

    """Model wrapper for TensorFlow models in SavedModel format"""
    def __init__(self, path=DEFAULT_MODEL_PATH):
        logger.info('Loading model from: {}...'.format(path))

        # load assets first to enable model definition
        self._load_assets(path)

        # Loading the tf SavedModel
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        tf.saved_model.loader.load(self.sess, [tag_constants.SERVING], DEFAULT_MODEL_PATH)

        self.word_ids_tensor = self.sess.graph.get_tensor_by_name('word_input:0')
        self.char_ids_tensor = self.sess.graph.get_tensor_by_name('char_input:0')
        self.output_tensor = self.sess.graph.get_tensor_by_name('predict_output/truediv:0')

    def _load_assets(self, path):
        vocab_tags = load_vocab(path + "/tags.txt")
        vocab_chars = load_vocab(path + "/chars.txt")
        vocab_words = load_vocab(path + "/words.txt")

        self.proc_fn = get_processing_word(vocab_words, vocab_chars, lowercase=True, chars=True)

        # Adding outside tag to the padding number
        self.id_to_tag = {idx: v for v, idx in vocab_tags.items()}
        dict_vocal_tags_len = len(vocab_tags)
        self.id_to_tag.update({dict_vocal_tags_len: 'O'})

        self.n_words = len(vocab_words)
        self.n_char = len(vocab_chars)
        n_tags = len(vocab_tags)
        self.pad_tag = n_tags
        # print('padding tags.....', self.pad_tag)
        self.n_labels = n_tags + 1

    def inter_process(self, words):
        word_ids = []
        char_ids = []
        for w in words:
            char_id, word_id = zip(*w)
            char_ids.append(char_id)
            word_ids.append(word_id)
        word_ids, _ = pad_sequences(word_ids, pad_tok=self.pad_tag)
        char_ids, _ = pad_sequences(char_ids, pad_tok=self.pad_tag, nlevels=2)
        word_ids_arr = np.array(word_ids)
        char_ids_arr = np.array(char_ids)
        return word_ids_arr, char_ids_arr

    # def _pre_process(self, x, predict_batch_size=2):
    #     print('---- Test -----')
    #     print(x)
    #     sentence_token = []
    #     for i in range(0, len(x), predict_batch_size):
    #         print(i)
    #         # Accumulate data
    #         input_data = x[i:i+predict_batch_size]
    #         # iterate through data and get sentence tokens
    #         words = []
    #
    #         for sentence in input_data:
    #             words_raw = re.split(self.pat, sentence)
    #             words_raw = [w.strip() for w in words_raw]      # strip whitespace
    #             words_raw = [w for w in words_raw if w]         # keep only non-empty terms, keeping raw punctuation
    #             words.append([self.proc_fn(w) for w in words_raw])
    #
    #             sentence_token.append(words_raw)
    #
    #
    #         # pad sentence
    #         print('batch words', words)
    #         word_ids_arr, char_ids_arr = self.inter_process(words)
    #     return sentence_token, word_ids_arr, char_ids_arr
    #
    # def _post_process(self, x):
    #     print('Inside post process')
    #     print(x)
    #     result = []
    #     for r in x:
    #         result.append([self.id_to_tag[i] for i in r.ravel()])
    #     print('final result', result)
    #     return result

    # def _predict(self, word_ids_arr, char_ids_arr):
    #     pred = self.sess.run(self.output_tensor, feed_dict={
    #         self.word_ids_tensor: word_ids_arr,
    #         self.char_ids_tensor: char_ids_arr
    #     })
    #     return np.argmax(pred, -1)

    def _predict(self, x, predict_batch_size=0):
        # sentence_token = []
        # result = []
        # pp_elapsed_time = []
        # inf_elapsed_time = []
        # total_inf_time = 0
        # each_inf_time = 0
        glob_empty = 0
        predict_batch_size = [ 2**j for j in range(3,10+1) ]
        total_inference_time_list = []

        for k in range(len(predict_batch_size)):

            sentence_token = []
            result = []
            pp_elapsed_time = []
            inf_elapsed_time = []
            total_inf_time = 0
            sentce_length = []
            ind_sort_sent = []

            for len_x in range(len(x)):
                sentce_length.append(len(x[len_x]))

            ind_sort_sent = np.argsort(sentce_length)
            new_index_order = ind_sort_sent.tolist()
            
            xx = []
            for x_ind in range (len(new_index_order)):
                xx.append(x[new_index_order[x_ind]])

            # ##################HERE IS MEMORY PEAK EVALUATION###########################
            # ######define run_options and run_metadata
            # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run_metadata = tf.RunMetadata() 
            # ##################HERE IS MEMORY PEAK EVALUATION###########################
            
            for i in range(0, len(xx), predict_batch_size[k]):
                #### x: whole documents with entire sentences
                #### x: list of many sentences
                #### x[0]: string

                inf_start_time =0 
                each_inf_time = 0

                # Accumulate data
                input_data =[]
                input_data = xx[i:i + predict_batch_size[k]]
                # iterate through data and get sentence tokens
                words = []

                #### input_data: a list includes 32 sentences ####
                #### input_data[0]: str ####
                
                pp_start_time = timeit.default_timer()
                range_batch_size = [ idx for idx in range(predict_batch_size[k])]
                if len(input_data) < predict_batch_size[k]: 
                    range_batch_size = [ idx for idx in range(len(input_data))]

                for i_sentence in range_batch_size:
                    # #### Using Fei data ####
                    # words_raw = re.split(self.pat, sentence)
                    # words_raw = [w.strip() for w in words_raw]  # strip whitespace
                    # words_raw = [w for w in words_raw if w]  # keep only non-empty terms, keeping raw punctuation
                    # #### Using Fei data ####

                    #### words_raw should be the input from Fei's data#######
                    #### words_raw is a list #######
                    #### words_raw[0] is a str; ex: < OR P #####
                    #### words is mapped to word number(ids) ######
                    
                    words_raw = input_data[i_sentence]
                    if len(input_data[i_sentence]) == 0:
                        words_raw = ['']
                        glob_empty += 1
                    
                    words.append([self.proc_fn(w) for w in words_raw])
                    # sentence_token.append(words_raw)

                # pad sentence          
                word_ids_arr, char_ids_arr = self.inter_process(words)
                pp_elapsed_time.append(timeit.default_timer() - pp_start_time)

                inf_start_time = timeit.default_timer()     
                pred = self.sess.run(self.output_tensor, feed_dict={
                    self.word_ids_tensor: word_ids_arr,
                    self.char_ids_tensor: char_ids_arr
                })

                # ##################HERE IS MEMORY PEAK EVALUATION###########################
                # pred = self.sess.run(self.output_tensor, feed_dict={
                #     self.word_ids_tensor: word_ids_arr,
                #     self.char_ids_tensor: char_ids_arr
                # }, options=run_options,
                # run_metadata=run_metadata)

                ##################HERE IS MEMORY PEAK EVALUATION###########################
                # ########## log the peak memory consumed by the model ##########
                # peak_memory = mem_util.peak_memory(run_metadata)
                # print("The peak memory consumed by the BiLSTM model: {} bytes".format(peak_memory))
                

                # mem_filename = 'memory_peak_Feb3rd_bts_' + str(predict_batch_size[k]) + '.txt'
                # with open(mem_filename, 'a') as fd:
                #     fd.write(str(peak_memory['/cpu:0'])+'\n')

                # ##########create the Timeline object, and write it to a json file##########
                # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                # chrome_trace = fetched_timeline.generate_chrome_trace_format()
                # # with open('profiling_trace_bts32.json', 'w') as file_handle:
                # #     file_handle.write(chrome_trace)
                # ##########create the Timeline object, and write it to a json file##########
                ##################HERE IS MEMORY PEAK EVALUATION###########################
                
                labels_pred_arr = np.argmax(pred, -1)
                # labels_pred_arr = tf.math.argmax(pred, -1)

                each_inf_time = timeit.default_timer() - inf_start_time
                total_inf_time += each_inf_time

                inf_elapsed_time.append(timeit.default_timer() - inf_start_time)

                # for r in labels_pred_arr:
                #     result.append([self.id_to_tag[i] for i in r.ravel()])


            ############## predicted labels #######################
            #labels_pred_arr = self._predict(word_ids_arr, char_ids_arr)
            #labels_pred = self._post_process(labels_pred_arr)
            ############## predicted labels #######################

            ####### No idea why sum / len ########
            # pp_elapsed_time.append(sum(pp_elapsed_time)/len(pp_elapsed_time))
            # inf_elapsed_time.append(sum(inf_elapsed_time)/len(inf_elapsed_time))
            ####### No idea why sum / len ########


            df = pd.DataFrame({'tokenization time': pp_elapsed_time,
                            'inference time': inf_elapsed_time, 
                            'total inf time': total_inf_time})

            InferTime_filename = '7Sorted_Inference_Time_bts_' + str(predict_batch_size[k]) + '.csv'
            df.to_csv(InferTime_filename)
            print('+++++++')
            print(total_inf_time)
            print('==========')
            print('predict_batch size', predict_batch_size[k])
            print('global empty word', glob_empty)
        
            total_inference_time_list.append(total_inf_time)
            print('all batch size inference time', total_inference_time_list)
        
        return result, total_inf_time

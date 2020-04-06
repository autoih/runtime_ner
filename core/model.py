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
        self.n_labels = n_tags + 1

    def _inter_process(self, words):
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

    
    def _Sentence_sorting(self, sentences):
        sentce_length = []
        ind_sort_sent = []
        sorted_sentences = []

        # get all length of all sentences
        for len_x in range(len(sentences)):
                sentce_length.append(len(sentences[len_x]))

        ind_sort_sent = np.argsort(sentce_length)
        new_index_order = ind_sort_sent.tolist()
            
        for x_ind in range (len(new_index_order)):
            sorted_sentences.append(sentences[new_index_order[x_ind]])
        
        return sorted_sentences

    def _word_id_process(self, input, range_of_btz):
        words = []
        for i_sent in range_of_btz:
            words_raw = input[i_sent]
            if len(input[i_sent]) == 0:
                words_raw = ['']
            words.append([self.proc_fn(w) for w in words_raw])

        return words


    def _process_each_batch(self, input_data, batch ):
        inf_start_time =0 
        each_inf_time = 0
        pp_time = 0

        # Deal with sentences of the last batch is smaller the batch size
        range_batch_size = [ idx for idx in range(batch)]
        if len(input_data) < batch: 
            range_batch_size = [ idx for idx in range(len(input_data))]

        pp_start_time = timeit.default_timer()
        # get word_id
        words = self._word_id_process(input_data, range_batch_size)
        # sentence_token.append(words_raw)

        # pad sentence          
        word_ids_arr, char_ids_arr = self._inter_process(words)
        pp_time = timeit.default_timer() - pp_start_time

        inf_start_time = timeit.default_timer()     
        pred = self.sess.run(self.output_tensor, feed_dict={
            self.word_ids_tensor: word_ids_arr,
            self.char_ids_tensor: char_ids_arr
        })      
        labels_pred_arr = np.argmax(pred, -1)

        each_inf_time = timeit.default_timer() - inf_start_time

        return each_inf_time, pp_time


    def _predict(self, x, predict_batch_size=0):

        predict_batch_size = [ 2**j for j in range(3,10+1) ]
        total_inference_time_list = []

        # run different batch sizes
        for k in range(len(predict_batch_size)):
            # sentence_token = []
            result = []
            pp_elapsed_time = []
            inf_elapsed_time = []
            total_inf_time = 0
            
            for document_ind in range(len(x)):
                one_doc = []
                one_doc = x[document_ind]

                Entire_sorted_sentence = self._Sentence_sorting(one_doc)
                # all batches in one batch size
                for i in range(0, len(Entire_sorted_sentence), predict_batch_size[k]):
                    # Accumulate data
                    sentence_batch =[]
                    sentence_batch = Entire_sorted_sentence[i:i + predict_batch_size[k]]
                    
                    each_inf_time, pp_time = self._process_each_batch(sentence_batch, predict_batch_size[k])
                    pp_elapsed_time.append(pp_time)
                    total_inf_time += each_inf_time
                    inf_elapsed_time.append(each_inf_time)

            # save each batch for entire sentence result
            df = pd.DataFrame({'tokenization time': pp_elapsed_time,
                            'inference time': inf_elapsed_time, 
                            'total inf time': total_inf_time})

            InferTime_filename = '10Sorted_Inference_Time_bts_' + str(predict_batch_size[k]) + '.csv'
            df.to_csv(InferTime_filename)
            total_inference_time_list.append(total_inf_time)

            print('total_inf_time: ', total_inf_time)
            print('predict_batch size: ', predict_batch_size[k])
            print('all batch size inference time: ', total_inference_time_list)
        
        return result, total_inf_time

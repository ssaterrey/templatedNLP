# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import nltk
import spacy
from pubsub import pub
import pysbd
import json
import numpy
import uuid
import time
import PySimpleGUI as sg


def levenshtein_distance(token1, token2):
    distances = numpy.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    a = 0
    b = 0
    c = 0

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if token1[t1 - 1] == token2[t2 - 1]:
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                if a <= b and a <= c:
                    distances[t1][t2] = a + 1
                elif b <= a and b <= c:
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len(token1)][len(token2)]


class Clause:
    def __init__(self, clause_info):
        self.sentence_id = clause_info['sid']
        self.complement_obj = {}
        self.complement_subj = {}
        self.subject = {}
        self.predicate = {}
        self.object_dir = {}
        self.object_in = {}
        self.adverbs = []
        self.missing = ["subject", "predicate", "complement_obj", "complement_subj", "object_dir", "object_in",
                        "adverb"]
        self.text = clause_info['text']
        self.interpretation = clause_info
        self.alternatives = []
        self.type = clause_info['type']
        self.is_info_request = False
        for elem in clause_info['template']:
            if elem['constituent'] in self.missing:
                self.missing.remove(elem['constituent'])
            if elem['constituent'] == "subject":
                self.subject = elem
            elif elem['constituent'] == "complement_obj":
                self.complement_obj = elem
            elif elem['constituent'] == "complement_subj":
                self.complement_subj = elem
            elif elem['constituent'] == "predicate":
                self.predicate = elem
            elif elem['constituent'] == "object_dir":
                self.object_dir = elem
            elif elem['constituent'] == "object_in":
                self.object_in = elem
            elif elem['constituent'] == "adverb":
                self.adverbs.append(elem)

    def add_alternative(self, alternative):
        self.alternatives.append(alternative)

    def get_constituent(self, constituent):
        if constituent == "subject":
            return self.subject
        elif constituent == "predicate":
            return self.predicate
        elif constituent == "objects":
            objects = []
            if self.object_dir:
                objects.append({"dir_object": self.object_dir})
            if self.complement_subj:
                objects.append({"indir_object": self.object_in})
            return objects
        elif constituent == "complement":
            complements = []
            if self.complement_obj:
                complements.append({"obj_complement": self.complement_obj})
            if self.complement_subj:
                complements.append({"subj_complement": self.complement_subj})
            return complements
        elif constituent == "adverbs":
            return self.adverbs

    def get_clause(self):
        return self.text

    def what_is_missing(self):
        return self.missing  # a list, possible: subject, predicate, object, complement, adverb, none

    def substitute(self, constituent, value):
        # objects and complements - allow for more details
        if constituent == "subject":
            self.subject = value
        elif constituent == "object_dir":
            self.object_dir = value
        elif constituent == "object_in":
            self.object_in = value
        elif constituent == "predicate":
            self.predicate = value
        elif constituent == "complement_obj":
            self.complement_obj = value
        elif constituent == "complement_subj":
            self.complement_subj = value
        elif constituent == "adverb":
            self.adverbs.append(value)


class Sentence:
    def __init__(self, message, key):
        self.sentence_id = key
        self.word_infos = []
        self.index = 0
        self.index2 = 0
        self.message = message
        self.interpretation = {}
        self.alternatives = []
        self.sentence_type = "question"
        self.is_info_request = False
        self.ready = False
        pub.subscribe(self.pos_feedback, 'pos_feedback')
        pub.subscribe(self.word_feedback, 'word_feedback')
        pub.subscribe(self.typo_feedback, 'typo_feedback')
        pub.subscribe(self.sentence_feedback, 'sentence_found')
        self.pre_parse()

    def is_ready(self):
        return self.ready

    # parses input sentence and stores results
    def pre_parse(self):
        # possible: declarative, imperative, exclamative, answer_template
        # send words one by one to the first level contexts
        # get words separately and alternative pos for them

        wordtags = nltk.ConditionalFreqDist((w.lower(), t)
                                            for w, t in nltk.corpus.brown.tagged_words(tagset="universal"))
        nlp = spacy.load('en_core_web_md')
        doc = nlp(self.message)
        for token in doc:
            # print(token)
            if token.dep_ == "punct":
                continue
            word_info = {}
            word_info['sid'] = self.sentence_id
            word_info['index'] = self.index
            self.index += 1
            # update everywhere for candidates
            word_info['candidates'] = []
            candidate = {}
            candidate['text'] = token.text
            candidate['pos_list'] = list(wordtags[token.text.lower()])  # if empty we need to run a typo check
            word_info['candidates'].append(candidate)
            self.word_infos.append(word_info)

        if self.word_infos:
            pass
        else:
            pub.sendMessage("parse_error", "Empty sentence!")

        self.index = len(self.word_infos)
        for wi in self.word_infos:
            for candidate in wi['candidates']:
                if candidate['pos_list']:
                    pub.sendMessage("pos", data=wi)
                else:
                    pub.sendMessage("typo", data=wi)

        # if any word is not recognized before sending to our context
        # run typo check (only 1 dist if any otherwise - "unknown word" - to the list of unknown)
        # send a list of possible candidates - contexts should pick the most suitable one -
        # both syntactically and semantically (entanglement, topic)

        # try to find words in our contexts
        # what we can find we update info, otherwise - to the list of unprocessed

        # when all typos are found without bigger errors,
        # when we replaced alternative info with ours,
        # only then we will be able to start parsing

    def parse(self):
        # send word one by one
        # wait for ack before sending another one
        # if any word was not accepted by at least one context - escalate "parse error - sentence"
        # print(self.word_infos)
        if self.index2 < len(self.word_infos):
            pub.sendMessage("word", data=self.word_infos[self.index2])
        else:
            pub.sendMessage("parsing_done", arg={"sid": self.sentence_id})

    def typo_feedback(self, word_info):
        # in case of typo we get a list of candidates - process properly
        if word_info['sid'] != self.sentence_id:
            return
        if word_info:
            for wi in self.word_infos:
                if wi['index'] == word_info['index']:
                    # replace one word with a list - maybe all of them should be lists
                    wi['candidates'] = word_info['candidates']
                    self.index -= 1
                    if self.index == 0:
                        self.index2 = 0
                        self.parse()
        else:
            # reflect this error somewhere so that we communicate it to user properly and continue chatting
            pub.sendMessage("parse_error", arg="Unknown word - " + word_info['candidates'][0]['text'])

    def word_feedback(self, word_info):
        if word_info['sid'] != self.sentence_id:
            return
        # print(word_info)
        if word_info['result'] == "err":
            pub.sendMessage("parse_error", arg="Dangling word - " + word_info['candidates'][0]['text'])
            # maybe, we need to cleanup
        elif word_info['result'] == "ack":
            self.index2 += 1
            self.parse()

    def pos_feedback(self, word_info):
        if word_info['sid'] != self.sentence_id:
            return
        for wi in self.word_infos:
            if wi['index'] == word_info['index']:
                wi['candidates'] = word_info['candidates']
                self.index -= 1
                if self.index == 0:
                    self.index2 = 0
                    self.parse()

    def sentence_feedback(self, sentence_info):
        if sentence_info['sid'] != self.sentence_id:
            return
        # if answer is expected - initiate a quest for it
        if self.index2 < len(self.word_infos) - 1:
            return

        if self.ready:
            self.alternatives.append(sentence_info)
        else:
            self.ready = True
            self.interpretation = sentence_info
            self.sentence_type = sentence_info["type"]

        # if self.sentence_type == "interrogative":  # self.is_info_request:
            # pub.sendMessage("find_info", arg=sentence_info)
        # else:
            # store clauses and conjunctions and sentence_id
        pub.sendMessage("store_info", sentence_info=sentence_info)

    def what_type(self):
        return self.sentence_type

    def is_info_request(self):
        return self.is_info_request

    def get_sentence(self):
        return self.message


class UserUI:
    def __init__(self, user_name):
        self.user_name = user_name
        pub.subscribe(self.msg_for_user, 'msg_for_user')
        pub.subscribe(self.process_err, "parse_error")

    def msg_for_user(self, arg):
        print("BOT: " + arg['sentence'])
        user_input = input()
        if user_input == "exit" or user_input == "stop" or user_input == "quit":
            quit()
        else:
            pub.sendMessage('parse_input', arg=user_input)

    def process_err(self, arg):
        pass


class Parser:
    def __init__(self):
        pub.subscribe(self.parse_input, 'parse_input')
        pub.subscribe(self.parse_next, 'parsing_done')
        self.index = 0
        self.sents = []

    def parse_input(self, arg):
        # print("Hello Parser")
        seg = pysbd.Segmenter(language="en", clean=False)
        if seg.segment(arg):
            self.sents = seg.segment(arg)
            self.index = 0
            key = str(uuid.uuid4())
            s = Sentence(self.sents[self.index], key)
            # we need to parse each sentence
            # and either store it or answer it (or show an error)
            # we want to do start working on next sentence after the previous one is done

    def parse_next(self, arg):
        self.index += 1
        if self.index < len(self.sents):
            key = str(uuid.uuid4())
            s = Sentence(self.sents[self.index], key)
        else:
            self.sents.clear()
            pub.sendMessage('msg_for_user', arg={'sentence': "Information has been processed."})


# pos
class ContextL0:
    def __init__(self, config):
        self.templates = {}
        self.config = config  # templates with id really
        self.nouns_dict = {}
        for word in self.config['nouns']:
            self.nouns_dict[word['id']] = word
        pub.subscribe(self.process_pos, 'pos')
        pub.subscribe(self.process_typo, 'typo')
        pub.subscribe(self.recognize_word, 'reco')

    # when we start a new sentence
    def clear(self):
        self.templates.clear()

    def process_pos(self, data):
        for word in self.config['words']:
            if data['candidates'][0]['text'] == word['id']:
                data['candidates'][0]['pos_list'] = word['POS']
                for pos in data['candidates'][0]['pos_list']:
                    if pos['pos'] == "NN":
                        pos['category'] = self.nouns_dict[pos['base_form']]['category']
                break
        for name in self.config['names']:
            if data['candidates'][0]['text'] == name['name']:
                data['candidates'][0]['pos_list'] = [{"pos": "NNP", "gender": name['gender'],
                                                      "number": "singular", "category": "name",
                                                      "base_form": name['name']}]
                break
        pub.sendMessage('pos_feedback', word_info=data)
        pass

    def process_typo(self, data):
        words = []
        for word in self.config['words']:
            if levenshtein_distance(data['candidates'][0]['text'], word['id']) == 1:
                word2 = {}
                word2['text'] = word['id']
                word2['pos_list'] = word['POS']
                words.append(word2)
        data['candidates'] = words
        pub.sendMessage('typo_feedback', word_info=data)

    # not only word, but also POS
    def recognize_word(self, data):
        pub.sendMessage('reco_feedback', arg=data)


# phrase contexts - all types (noun, adjective, adverb) - not verbs
class ContextL1:
    def __init__(self, config):
        self.templates = {}
        self.config = config  # templates with id really
        pub.subscribe(self.process_word, 'word')

    # when we start a new sentence
    def clear(self):
        self.templates.clear()

    def process_word(self, data):
        if data['index'] == 0:
            self.templates.clear()
        # either 'ack' or 'err'
        # we have templates with optional or possibly omitted parts
        # we go through each of them for each word or candidate
        # if any template picked up the pos we update flag
        # if any template is completed we send a signal
        # complete phrases are processed internally, not through pubsub
        # print(data)

        check = True
        sentence_check = False
        completed_templates = []
        # go through blank templates - if this type is expected
        # at the first obligatory position or any optional position
        # before it then we create a new candidate - it gets id,
        # stores template id and resulting type and index of element filled so far,
        # and indexes of elements from the sentence
        for template in self.config["phrase_templates"]:
            ind = 0
            check2 = True
            while ind < len(template['template']) and template['template'][ind]['flag'] == 'optional':
                for obj in template['template'][ind]['obj_id']:
                    for cand in data['candidates']:
                        for pos in cand['pos_list']:
                            if pos['pos'] == obj:
                                if obj[0:2] == "VB" and 'verb_req' in template['template'][ind] and \
                                        template['template'][ind]['verb_req'] != pos['verb_type']:
                                    continue
                                # create new template
                                new_template = {}
                                new_template['sid'] = data["sid"]
                                new_template['tid'] = template['tid']
                                new_template['start_index'] = data['index']
                                new_template['end_index'] = data['index']
                                new_template['template_index'] = ind
                                new_template['template'] = template['template']
                                new_template['template_result'] = template['template_result']
                                new_template['elems'] = []
                                el = {'pos': pos, 'text': cand['text']}
                                if "constituent" in obj:
                                    el["constituent"] = obj["constituent"]
                                new_template['elems'].append(el)
                                if template['tid'][0:6] == 'clause':
                                    new_template['formula'] = template['formula']
                                    new_template['type'] = template['type']
                                    if "answer_template" in template:
                                        new_template['answer_template'] = template['answer_template']
                                key = str(uuid.uuid4())
                                self.templates[key] = new_template
                                # print(new_template)
                                check2 = False
                                check = False
                                ind2 = ind + 1
                                check3 = False
                                while ind2 < len(template['template']):
                                    if template['template'][ind2]['flag'] == 'obligatory':
                                        check3 = True
                                        break
                                    ind2 += 1
                                if not check3 or ind == len(template['template']) - 1:
                                    completed_templates.append(new_template)
                                    if new_template['tid'][0:4] == 'sent' and new_template['start_index'] == 0:
                                        pub.sendMessage('sentence_found', sentence_info=new_template)
                                break
                        if not check2:
                            break
                    if not check2:
                        break
                if not check2:
                    break
                ind += 1
            if ind < len(template['template']) and template['template'][ind]['flag'] == 'obligatory' and check2:
                # print(template['template'][ind]['obj_id'])
                for obj in template['template'][ind]['obj_id']:
                    for cand in data['candidates']:
                        # print(cand['pos_list'])
                        for pos in cand['pos_list']:
                            if pos['pos'] == obj:
                                if obj[0:2] == "VB" and 'verb_req' in template['template'][ind] and \
                                        template['template'][ind]['verb_req'] != pos['verb_type']:
                                    continue
                                # create new template
                                new_template = {}
                                new_template['sid'] = data["sid"]
                                new_template['tid'] = template['tid']
                                new_template['start_index'] = data['index']
                                new_template['end_index'] = data['index']
                                new_template['template_index'] = ind
                                new_template['template'] = template['template']
                                new_template['template_result'] = template['template_result']
                                new_template['elems'] = []
                                el = {'pos': pos, 'text': cand['text']}
                                if "constituent" in template['template'][ind]:
                                    el["constituent"] = template['template'][ind]["constituent"]
                                new_template['elems'].append(el)
                                if template['tid'][0:6] == 'clause':
                                    new_template['formula'] = template['formula']
                                    new_template['type'] = template['type']
                                    if "answer_template" in template:
                                        new_template['answer_template'] = template['answer_template']
                                key = str(uuid.uuid4())
                                self.templates[key] = new_template
                                # print(new_template)
                                check = False
                                ind2 = ind + 1
                                check3 = False
                                while ind2 < len(template['template']):
                                    if template['template'][ind2]['flag'] == 'obligatory':
                                        check3 = True
                                        break
                                    ind2 += 1
                                if not check3 or ind == len(template['template']) - 1:
                                    completed_templates.append(new_template)
                                    if new_template['tid'][0:4] == 'sent' and new_template['start_index'] == 0:
                                        pub.sendMessage('sentence_found', sentence_info=new_template)
                                break
                        if not check2:
                            break
                    if not check2:
                        break
        # go through filled templates
        for template in self.templates.values():
            ind = template['template_index'] + 1
            check2 = True
            # print(template)
            if template['end_index'] == data['index'] - 1:
                while ind < len(template['template']) and template['template'][ind]['flag'] == 'optional':
                    for obj in template['template'][ind]['obj_id']:
                        for cand in data['candidates']:
                            for pos in cand['pos_list']:
                                if pos['pos'] == obj:
                                    if obj[0:2] == "VB" and 'verb_req' in template['template'][ind] and \
                                            template['template'][ind]['verb_req'] != pos['verb_type']:
                                        continue
                                    # update template
                                    template['end_index'] = data['index']
                                    template['template_index'] = ind
                                    el = {'pos': pos, 'text': cand['text']}
                                    if "constituent" in template['template'][ind]:
                                        el["constituent"] = template['template'][ind]["constituent"]
                                    template['elems'].append(el)
                                    # print(template)
                                    check2 = False
                                    check = False
                                    ind2 = ind + 1
                                    check3 = False
                                    while ind2 < len(template['template']):
                                        if template['template'][ind2]['flag'] == 'obligatory':
                                            check3 = True
                                            break
                                        ind2 += 1
                                    if not check3 or ind == len(template['template']) - 1:
                                        completed_templates.append(template)
                                        if template['tid'][0:4] == 'sent' and template['start_index'] == 0:
                                            pub.sendMessage('sentence_found', sentence_info=template)
                                    break
                            if not check2:
                                break
                        if not check2:
                            break
                    if not check2:
                        break
                    ind += 1
            if ind < len(template['template']) and template['template'][ind]['flag'] == 'obligatory' and check2 \
                    and template['end_index'] == data['index'] - 1:
                for obj in template['template'][ind]['obj_id']:
                    for cand in data['candidates']:
                        for pos in cand['pos_list']:
                            if pos['pos'] == obj:
                                if obj[0:2] == "VB" and 'verb_req' in template['template'][ind] and \
                                        template['template'][ind]['verb_req'] != pos['verb_type']:
                                    continue
                                # update new template
                                template['end_index'] = data['index']
                                template['template_index'] = ind
                                el = {'pos': pos, 'text': cand['text']}
                                if "constituent" in template['template'][ind]:
                                    el["constituent"] = template['template'][ind]["constituent"]
                                template['elems'].append(el)
                                # print(template)
                                check = False
                                ind2 = ind + 1
                                check3 = False
                                while ind2 < len(template['template']):
                                    if template['template'][ind2]['flag'] == 'obligatory':
                                        check3 = True
                                        break
                                    ind2 += 1
                                if not check3 or ind == len(template['template']) - 1:
                                    completed_templates.append(template)
                                    if template['tid'][0:4] == 'sent' and template['start_index'] == 0:
                                        pub.sendMessage('sentence_found', sentence_info=template)
                                break
                        if not check2:
                            break
                    if not check2:
                        break
        while completed_templates:
            completed_templates2 = []
            # print(completed_templates)
            for temp in completed_templates:
                # go through each raw template if any requires this one from start
                # go through each started template if any requires this one from current position
                for template in self.config["phrase_templates"]:
                    ind = 0
                    check2 = True
                    while ind < len(template['template']) and template['template'][ind]['flag'] == 'optional':
                        for obj in template['template'][ind]['obj_id']:
                            if temp['template_result'] == obj:
                                # create new template
                                new_template = {}
                                new_template['sid'] = temp["sid"]
                                new_template['tid'] = template['tid']
                                new_template['start_index'] = temp['start_index']
                                new_template['end_index'] = temp['end_index']
                                new_template['template_index'] = ind
                                new_template['template'] = template['template']
                                new_template['template_result'] = template['template_result']
                                new_template['elems'] = []
                                el = temp
                                if "constituent" in template['template'][ind]:
                                    el["constituent"] = template['template'][ind]["constituent"]
                                new_template['elems'].append(el)
                                if template['tid'][0:6] == 'clause':
                                    new_template['formula'] = template['formula']
                                    new_template['type'] = template['type']
                                    if "answer_template" in template:
                                        new_template['answer_template'] = template['answer_template']
                                if template['tid'][0:4] == 'sent':
                                    if temp['tid'][0:6] == 'clause':
                                        new_template['type'] = temp['type']
                                        if "answer_template" in temp:
                                            new_template['answer_template'] = temp['answer_template']
                                key = str(uuid.uuid4())
                                self.templates[key] = new_template
                                # print(new_template)
                                check2 = False
                                ind2 = ind + 1
                                check3 = False
                                while ind2 < len(template['template']):
                                    if template['template'][ind2]['flag'] == 'obligatory':
                                        check3 = True
                                        break
                                    ind2 += 1
                                if not check3 or ind == len(template['template']) - 1:
                                    completed_templates2.append(new_template)
                                    if new_template['tid'][0:4] == 'sent' and new_template['start_index'] == 0:
                                        pub.sendMessage('sentence_found', sentence_info=new_template)
                                break
                        if not check2:
                            break
                        ind += 1
                    if ind < len(template['template']) and template['template'][ind]['flag'] == 'obligatory' and check2:
                        for obj in template['template'][ind]['obj_id']:
                            if temp['template_result'] == obj:
                                # create new template
                                new_template = {}
                                new_template['sid'] = temp["sid"]
                                new_template['tid'] = template['tid']
                                new_template['start_index'] = temp['start_index']
                                new_template['end_index'] = temp['end_index']
                                new_template['template_index'] = ind
                                new_template['template'] = template['template']
                                new_template['template_result'] = template['template_result']
                                new_template['elems'] = []
                                el = temp
                                if "constituent" in template['template'][ind]:
                                    el["constituent"] = template['template'][ind]["constituent"]
                                new_template['elems'].append(el)
                                if template['tid'][0:6] == 'clause':
                                    new_template['formula'] = template['formula']
                                    new_template['type'] = template['type']
                                    if "answer_template" in template:
                                        new_template['answer_template'] = template['answer_template']
                                if template['tid'][0:4] == 'sent':
                                    if temp['tid'][0:6] == 'clause':
                                        new_template['type'] = temp['type']
                                        if "answer_template" in temp:
                                            new_template['answer_template'] = temp['answer_template']
                                key = str(uuid.uuid4())
                                self.templates[key] = new_template
                                # print(new_template)
                                ind2 = ind + 1
                                check3 = False
                                while ind2 < len(template['template']):
                                    if template['template'][ind2]['flag'] == 'obligatory':
                                        check3 = True
                                        break
                                    ind2 += 1
                                if not check3 or ind == len(template['template']) - 1:
                                    completed_templates2.append(new_template)
                                    if new_template['tid'][0:4] == 'sent' and new_template['start_index'] == 0:
                                        pub.sendMessage('sentence_found', sentence_info=new_template)
                                break
                # go through filled templates
                for template in self.templates.values():
                    ind = template['template_index'] + 1
                    check2 = True
                    # print(template)
                    if template['end_index'] == temp['start_index'] - 1:
                        while ind < len(template['template']) and template['template'][ind]['flag'] == 'optional':
                            for obj in template['template'][ind]['obj_id']:
                                if temp['template_result'] == obj:
                                    # update template
                                    template['end_index'] = temp['end_index']
                                    template['template_index'] = ind
                                    el = temp
                                    if "constituent" in template['template'][ind]:
                                        el["constituent"] = template['template'][ind]["constituent"]
                                    template['elems'].append(el)
                                    # print(template)
                                    check2 = False
                                    ind2 = ind + 1
                                    check3 = False
                                    while ind2 < len(template['template']):
                                        if template['template'][ind2]['flag'] == 'obligatory':
                                            check3 = True
                                            break
                                        ind2 += 1
                                    if not check3 or ind == len(template['template']) - 1:
                                        completed_templates2.append(template)
                                        if template['tid'][0:4] == 'sent' and template['start_index'] == 0:
                                            pub.sendMessage('sentence_found', sentence_info=template)
                                    break
                            if not check2:
                                break
                            ind += 1
                    if ind < len(template['template']) and template['template'][ind]['flag'] == 'obligatory' \
                            and check2 and template['end_index'] == temp['start_index'] - 1:
                        for obj in template['template'][ind]['obj_id']:
                            if temp['template_result'] == obj:
                                # update template
                                template['end_index'] = temp['end_index']
                                template['template_index'] = ind
                                el = temp
                                if "constituent" in template['template'][ind]:
                                    el["constituent"] = template['template'][ind]["constituent"]
                                template['elems'].append(el)
                                # print(template)
                                ind2 = ind + 1
                                check3 = False
                                while ind2 < len(template['template']):
                                    if template['template'][ind2]['flag'] == 'obligatory':
                                        check3 = True
                                        break
                                    ind2 += 1
                                if not check3 or ind == len(template['template']) - 1:
                                    completed_templates2.append(template)
                                    if template['tid'][0:4] == 'sent' and template['start_index'] == 0:
                                        pub.sendMessage('sentence_found', sentence_info=template)
                                break
            completed_templates = completed_templates2
        # if a given word was not picked up - send an "err"
        if check:
            data['result'] = "err"
            pub.sendMessage('word_feedback', word_info=data)
        else:
            # no error
            # send "ack"
            data['result'] = "ack"
            pub.sendMessage('word_feedback', word_info=data)


# clause contexts - accepts also verbs as connecting pillar
# processes also coordination based on roles
# roles within a template are assumed based on the template expectations and incoming phrase
# for now - it may coordinate clauses in a sentence finally
class ContextL2:
    def __init__(self, config):
        self.templates = []
        self.config = config  # templates with id really
        # pub.subscribe(self.process_word, 'word')
        # pub.subscribe(self.process_phrase, 'phrase_found')

    # when we start a new sentence
    def clear(self):
        self.templates.clear()

    def process_word(self, data):
        # either 'ack' or 'err'
        check = True
        for template in self.templates:
            pass
        for template in self.config["verb_templates"]:
            pass
        pub.sendMessage('word_feedback', word_info=data)
        pub.sendMessage('clause_found', data=data)
        pass

    def process_phrase(self, data):
        pub.sendMessage('clause_found', data=data)
        pass


# sentence contexts - accepts clauses and conjunctions
class ContextL3:
    def __init__(self, config):
        self.templates = []
        self.config = config  # templates with id really
        # pub.subscribe(self.process_word, 'word')
        # pub.subscribe(self.process_clause, 'clause_found')

    # when we start a new sentence
    def clear(self):
        self.templates.clear()

    def process_word(self, data):
        # either 'ack' or 'err'
        pass
        pub.sendMessage('word_feedback', arg={'user_input': data})
        pub.sendMessage('sentence_found', arg={'user_input': data})

    def process_clause(self, data):
        pass
        # pub.sendMessage('sentence_found', arg={'user_input': data})


# add methods for organizing time objects
# consider moving this functionality into Memory class to make direct calls
class TimeCurator:
    def __init__(self):
        self.timestamps = {}
        pub.subscribe(self.get_timestamp, 'get_timestamp')

    def get_timestamp(self, arg):
        key = str(uuid.uuid4())
        time_object = {}
        time_object['sentence_id'] = arg['sentence_id']
        time_object['range'] = arg['range']
        time_object['timestamp'] = arg['timestamp']
        time_object['ref_time'] = arg['ref_time']
        self.timestamps[key] = time_object
        arg['timestamp'] = key
        pub.sendMessage('timestamp', arg=arg)


# sentence contexts - accepts clauses and conjunctions
class Memory:
    def __init__(self, config):
        self.configs = config
        self.nouns_dict = {}
        for word in self.configs['nouns']:
            self.nouns_dict[word['id']] = word
        self.adjs_dict = {}
        for word in self.configs['adjectives']:
            self.adjs_dict[word['id']] = word
        self.verbs_dict = {}
        for word in self.configs['verbs']:
            self.verbs_dict[word['id']] = word
        self.names_dict = {}
        for word in self.configs['names']:
            self.names_dict[word['name']] = word
        self.records = {}  # get from JSON file for now
        self.questions = {}
        self.answers = {}
        self.timestamps = {}
        self.objects = {}
        self.mentions = []
        self.properties = {}
        pub.subscribe(self.retrieve, 'find_info')
        pub.subscribe(self.process_sentence, 'store_info')

    def get_timestamp(self, arg):
        key = str(uuid.uuid4())
        time_object = {}
        time_object['sid'] = arg['sid']
        named_tuple = time.localtime()  # get struct_time
        # time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
        # result = time.strptime(time_string, "%m/%d/%Y, %H:%M:%S")
        time_object['time'] = named_tuple
        self.timestamps[key] = time_object
        arg['timestamp'] = key
        return arg

    def is_match(self, val1, val2, pos):
        if pos == "NNP":
            if val1 == val2:
                return "yav"
            name_info = self.names_dict[val2]
            for syn in name_info['synonyms']:
                if val1 == syn:
                    return "yav"
        if pos == "NN":
            if val1 == val2:
                return "yav"
            word_info = self.nouns_dict[val2]['meanings'][0]
            for syn in word_info['synonyms']:
                if val1 == syn:
                    return "yav"
            for ant in word_info['antonyms']:
                if val1 == ant:
                    return "nav"
            for hi_gen in word_info['hi_generalizations']:
                if val1 == hi_gen:
                    return "yav"
        if pos == "JJ":
            if val1 == val2:
                return "yav"
            word_info = self.adjs_dict[val2]['meanings'][0]
            for syn in word_info['synonyms']:
                if val1 == syn:
                    return "yav"
            for ant in word_info['antonyms']:
                if val1 == ant:
                    return "nav"
            for hi_gen in word_info['hi_generalizations']:
                if val1 == hi_gen:
                    return "yav"
        if pos == "VB":
            if val1 == val2:
                return "yav"
            word_info = self.verbs_dict[val2]['meanings'][0]
            for syn in word_info['synonyms']:
                if val1 == syn:
                    return "yav"
            for ant in word_info['antonyms']:
                if val1 == ant:
                    return "nav"
            for hi_gen in word_info['hi_generalizations']:
                if val1 == hi_gen:
                    return "yav"
        return "prav"

    # if there is no suitable, create one
    def check_ref(self, elem, time_ref):
        # extract NNOM
        nnom = {}
        for el in elem['elems']:
            if 'template_result' in el and el['template_result'] == "NNOM":
                nnom = el
        adj = {}
        noun = {}
        ng = {}
        if 'elems' in nnom:
            for el in nnom['elems']:
                if 'template_result' in el and el['template_result'] == "AdjP":
                    adj = el['elems'][0]
                elif 'template_result' in el and el['template_result'] == "NG":
                    ng = el
                else:
                    noun = el

        # recognize words for each obj in the objects

        # {
        #     "obj_id": "28c0c941-c0f7-4784-b00c-62ae4fa70eb3",
        #     'gender': 'both',
        #     'number': 'singular',
        #     'owner': 'obj_id',
        #     "pairs": [
        #         {"sid": "", "timestamp": "", "property": "", "value": ""}
        #     ],
        #     "triplets": [
        #         {"sid": "", "timestamp": "", "property": "", "action": "", "role": ""}
        #     ],
        # }

        ng_ref = ""
        match_found = False
        if ng:
            category = ng['category']
            val = ng['pos']['base_form']
            for key in self.objects:
                obj = self.objects[key]
                if match_found:
                    break
                for pair in obj['pairs']:
                    if pair['property'] != category:
                        continue
                    if self.is_match(val, pair['value'], ng['pos']['pos']) == "yav":
                        ng_ref = obj['obj_id']
                        match_found = True

        ref_obj_id = ""
        assessment = 0
        match_found = False

        for key in self.objects:
            obj = self.objects[key]
            if ng:
                if obj['owner'] and ng_ref != obj['owner']:
                    continue

            if adj:
                adj_ok = False
                val = adj['pos']['base_form']
                category = self.adjs_dict[val]['meanings'][0]['property']
                if 'pairs' in obj:
                    for pair in obj['pairs']:
                        if pair['property'] != category:
                            continue
                        if self.is_match(val, pair['value'], 'JJ') == "yav":
                            adj_ok = True
                            break
                if not adj_ok:
                    continue

            # check noun
            category = noun['pos']['category']
            val = noun['pos']['base_form']

            if 'pairs' in obj:
                for pair in obj['pairs']:
                    if pair['property'] != category:
                        continue
                    if self.is_match(val, pair['value'], noun['pos']['pos']) == "yav":
                        ref_obj_id = obj['obj_id']
                        match_found = True
            if match_found:
                break

        if not match_found:
            ref_obj_id = str(uuid.uuid4())
            object = {}
            object["obj_id"] = ref_obj_id
            object['time_ref'] = time_ref
            object["gender"] = noun['pos']["gender"]
            object["number"] = noun['pos']["number"]
            object["owner"] = ""
            if ng_ref:
                object["owner"] = ng_ref
            object["pairs"] = []
            category = noun['pos']['category']
            val = noun['pos']['base_form']
            pair = {}
            pair['sid'] = elem['sid']
            pair['time_ref'] = time_ref
            pair['property'] = category
            pair['value'] = val
            object["pairs"].append(pair)
            object["triplets"] = []
            self.objects[ref_obj_id] = object
            # print(object)
        else:
            if ng and not obj['owner']:
                obj['owner'] = ng_ref

        return ref_obj_id

    def assign_complement_ref(self, arg, time_ref):
        el = arg['elems'][0]
        obj_ref = ""

        if 'template_result' in el and el['template_result'] == "NP":
            obj_ref = self.check_ref(el, time_ref)
        elif 'template_result' in el and el['template_result'] == "PronP":
            obj_ref = self.check_pron_ref(el, time_ref)

        arg['reference_obj_id'] = obj_ref
        self.mentions.append(obj_ref)
        complement = arg['elems'][2]
        complement['reference_obj_id'] = obj_ref
        arg['elems'][2]['reference_obj_id'] = obj_ref
        if 'template_result' in complement and complement['template_result'] == "NP":
            nnom = {}
            for el2 in complement['elems']:
                if 'template_result' in el2 and el2['template_result'] == "NNOM":
                    nnom = el2
            adj = {}
            noun = {}
            ng = {}
            if 'elems' in nnom:
                for el2 in nnom['elems']:
                    if 'template_result' in el2 and el2['template_result'] == "AdjP":
                        adj = el2['elems'][0]
                    elif 'template_result' in el2 and el2['template_result'] == "NG":
                        ng = el2
                    else:
                        noun = el2
            if adj:
                val = adj['pos']['base_form']
                category = self.adjs_dict[val]['meanings'][0]['property']
                pair = {}
                pair['sid'] = el['sid']
                pair['time_ref'] = time_ref
                pair['property'] = category
                pair['value'] = val
                self.objects[obj_ref]['pairs'].append(pair)
            category = noun['pos']['category']
            val = noun['pos']['base_form']
            pair = {}
            pair['sid'] = el['sid']
            pair['time_ref'] = time_ref
            pair['property'] = category
            pair['value'] = val
            self.objects[obj_ref]['pairs'].append(pair)

        if 'template_result' in complement and complement['template_result'] == "AdjP":
            adj = complement['elems'][0]
            val = adj['pos']['base_form']
            category = self.adjs_dict[val]['meanings'][0]['property']
            pair = {}
            pair['sid'] = el['sid']
            pair['time_ref'] = time_ref
            pair['property'] = category
            pair['value'] = val
            self.objects[obj_ref]['pairs'].append(pair)

        return arg

    # must be, find first suitable
    def check_pron_ref(self, elem, time_ref):
        # run back in time and find the first one that matches gender and number and role
        ref_obj_id = ""
        # print(elem)
        gender = elem['elems'][0]['pos']['gender']
        number = elem['elems'][0]['pos']['number']
        for i in reversed(self.mentions):
            obj = self.objects[i]
            gen = obj['gender']
            num = obj['number']
            if gender == gen and number == num:
                ref_obj_id = obj['obj_id']
                # check action
                break
            if gender == "he":
                if gen == "she" or gen == "it":
                    continue
            if gender == "she":
                if gen == "he" or gen == "it":
                    continue
            if gender == "it":
                if gen == "she" or gen == "he":
                    continue
            if gender == "both":
                if gen == "it":
                    continue

            if number == "singular":
                if num == "plural":
                    continue
            if number == "plural":
                if num == "singular":
                    continue

            ref_obj_id = obj['obj_id']
            # check action
            break

        return ref_obj_id

    def assign_obj_ref(self, arg, time_ref):
        # one thing - recognize objects
        # another thing - add info from complements as additional refs
        if "formula" in arg and arg['formula'] == "SVC":
            arg = self.assign_complement_ref(arg, time_ref)
        if 'reference_obj_id' not in arg and 'template_result' in arg and arg['template_result'] == 'NP':
            arg['reference_obj_id'] = self.check_ref(arg, time_ref)
            self.mentions.append(arg['reference_obj_id'])
            # print(arg)
        if 'template_result' in arg and arg['template_result'] == 'PronP':
            arg['reference_obj_id'] = self.check_pron_ref(arg, time_ref)
            self.mentions.append(arg['reference_obj_id'])
        # update list value in for loop
        # for i, s in enumerate(a):
        # a[i] = s.strip()
        if 'elems' in arg:
            for i, s in enumerate(arg['elems']):
                arg['elems'][i] = self.assign_obj_ref(s, time_ref)
        return arg

    def process_sentence(self, sentence_info):
        sentence_info = self.get_timestamp(sentence_info)
        sentence_info = self.assign_obj_ref(sentence_info, sentence_info['timestamp'])
        # print("Sentence stored: " + str(sentence_info))
        # print("Objects: " + str(self.objects))
        if sentence_info['type'] == 'interrogative':  # is_information_request():
            # retrieve answer and send response
            self.questions[sentence_info['sid']] = sentence_info
            with open("memory_questions.json", "w") as write_file:
                json.dump(self.questions, write_file)
            ans = self.retrieve(sentence_info)
            pub.sendMessage('response_retrieved', arg=ans)
        else:
            # store
            self.records[sentence_info['sid']] = sentence_info
            with open("memory_records.json", "w") as write_file:
                json.dump(self.records, write_file)
            # pub.sendMessage('sentence_stored', arg=sentence_info)

    # search carefully when a constituent is represented by a multi-word phrase
    # it is not just about the presence of that word in the phrase
    def retrieve(self, arg):
        check = True
        answer = {}
        key = str(uuid.uuid4())
        answer['sid'] = key
        answer['template'] = arg['elems'][0]['answer_template']
        answer['elems'] = []
        answer['text'] = ""
        assessment = 0
        subj = {}
        vb = {}
        for el in arg['elems'][0]['elems']:
            if el['constituent'] == "subject":
                subj = el
            elif el['constituent'] == "verb":
                vb = el
        for rec in self.records.values():
            if arg['elems'][0]['tid'] == "clause_2":
                subj_rec = {}
                vb_rec = {}
                for el in rec['elems'][0]['elems']:
                    if el['constituent'] == "subject":
                        subj_rec = el
                    elif el['constituent'] == "verb":
                        vb_rec = el
                subj_ref = ""
                if subj['template_result'] == "NP":
                    subj_ref = self.check_ref(subj, arg['timestamp'])
                else:
                    subj_ref = self.check_pron_ref(subj, arg['timestamp'])
                rec_ref = ""
                if subj_rec['template_result'] == "NP":
                    rec_ref = self.check_ref(subj_rec, rec['timestamp'])
                else:
                    rec_ref = self.check_pron_ref(subj_rec, rec['timestamp'])
                if subj_ref != rec_ref:
                    continue
                rez = self.is_match(vb['pos']['base_form'], vb_rec['elems'][0]['pos']['base_form'], "VB")
                if rez == "prav":
                    continue
                gen = ""
                if subj['template_result'] == "NP":
                    for el in subj['elems']:
                        if el['template_result'] == "NNOM":
                            for el2 in el['elems']:
                                if 'pos' in el2:
                                    gen = el2['pos']['gender']
                else:
                    gen = subj['elems'][0]['pos']['gender']
                if rez == "yav":
                    answer = {'text': "Yes, " + gen + " " + arg['elems'][0]['elems'][0]['text']}
                elif rez == "nav":
                    answer = {'text': "No, " + gen + " " + arg['elems'][0]['elems'][0]['text'] + " not"}
                check = False
                break
            elif arg['elems'][0]['tid'] == "clause_3":
                subj_rec = {}
                vb_rec = {}
                for el in rec['elems'][0]['elems']:
                    if el['constituent'] == "subject":
                        subj_rec = el
                    elif el['constituent'] == "verb":
                        vb_rec = el

                if self.is_match(vb['pos']['base_form'], vb_rec['elems'][0]['pos']['base_form'], "VB") != "yav":
                    continue
                rec_ref = ""
                np = ""
                if subj_rec['template_result'] == "NP":
                    rec_ref = self.check_ref(subj_rec, rec['timestamp'])
                    # print(subj_rec)
                    np = self.objects[rec_ref]['pairs'][0]['value']
                    # for el in subj_rec['elems']:
                    #     # print(el)
                    #     if 'template_result' in el and el['template_result'] == "NNOM":
                    #         for el2 in el['elems']:
                    #             if 'pos' in el2:
                    #                 np = el2['text']
                else:
                    rec_ref = self.check_pron_ref(subj_rec, rec['timestamp'])
                    np = self.objects[rec_ref]['pairs'][0]['value']
                # find proper form of aux verb
                v_aux = ""
                if arg['elems'][0]['elems'][1]['pos']['pos'][0:1] == "A":
                    v_aux = arg['elems'][1]['text']
                elif vb['pos']['pos'] == "VB":
                    v_aux = "do"
                elif vb['pos']['pos'] == "VBD":
                    v_aux = "did"
                elif vb['pos']['pos'] == "VBZ":
                    v_aux = "does"
                answer = {'text': np + " " + v_aux}
                check = False
                break
        if check:
            answer = {'text': "I do not know."}

        return answer['text']


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


class SampleGUI:
    def __init__(self):
        sg.theme('GreenTan')
        self.layout = [[sg.Multiline(size=(80, 20), reroute_stdout=True, echo_stdout_stderr=True)],
                  [sg.MLine(size=(70, 5), key='-MLINE IN-', enter_submits=True, do_not_clear=False),
                   sg.Button('SEND', bind_return_key=True), sg.Button('EXIT')]]

        self.window = sg.Window('Chat Window', self.layout, default_element_size=(30, 2), finalize=True)

        pub.subscribe(self.msg_for_user, 'msg_for_user')
        pub.subscribe(self.response_retrieved, 'response_retrieved')

    def response_retrieved(self, arg):
        print("BOT: " + arg)

    def msg_for_user(self, arg):
        print("BOT: " + arg['sentence'])

    def run(self):
        print("How can I help you?")
        while True:             # Event Loop
            self.event, self.values = self.window.read()
            if self.event in (sg.WIN_CLOSED, 'EXIT'):
                break

            if self.event == 'SEND':
                string = self.values['-MLINE IN-'].rstrip()
                print('  ' + string)
                pub.sendMessage('parse_input', arg=string)

        self.window.close()


def main():
    # print("Welcome to Project Alan!")

    # read json files and provide configs
    words = []
    adjectives = []
    adverbs = []
    conjunctions = []
    determiners = []
    nouns = []
    numbers = []
    prepositions = []
    pronouns = []
    verbs = []
    names = []

    with open("configs/names.json", "r", encoding="utf-8") as read_file:
        names = json.load(read_file)

    with open("configs/words_pos.json", "r") as read_file:
        words = json.load(read_file)
    with open("configs/adjectives_use.json", "r") as read_file:
        adjectives = json.load(read_file)
    with open("configs/adverbs_use.json", "r") as read_file:
        adverbs = json.load(read_file)
    with open("configs/conjunctions_use.json", "r") as read_file:
        conjunctions = json.load(read_file)
    with open("configs/determiners_use.json", "r") as read_file:
        determiners = json.load(read_file)
    with open("configs/nouns_use.json", "r") as read_file:
        nouns = json.load(read_file)
    with open("configs/pronouns_use.json", "r") as read_file:
        pronouns = json.load(read_file)
    with open("configs/prepositions_use.json", "r") as read_file:
        prepositions = json.load(read_file)
    with open("configs/verbs_use.json", "r") as read_file:
        verbs = json.load(read_file)

    arg0 = {"words": words, "adjectives": adjectives, "adverbs": adverbs,
            "conjunctions": conjunctions, "determiners": determiners,
            "nouns": nouns, "names": names, "pronouns": pronouns,
            "prepositions": prepositions, "verbs": verbs}

    level0 = ContextL0(arg0)

    phrases = []
    phrase_templates = []

    with open("configs/phrases_pos.json", "r") as read_file:
        phrases = json.load(read_file)
    with open("configs/phrase_templates.json", "r") as read_file:
        phrase_templates = json.load(read_file)

    arg1 = {"phrases": phrases, "phrase_templates": phrase_templates}

    level1 = ContextL1(arg1)

    verb_templates = []

    with open("configs/verb_templates.json", "r") as read_file:
        verb_templates = json.load(read_file)

    arg2 = {"verb_templates": verb_templates}

    level2 = ContextL2(arg2)

    arg3 = {}

    level3 = ContextL3(arg3)

    # mem = {}

    memory = Memory(arg0)

    # pub.subscribe(send_message, 'parse_sentence')
    parser = Parser()

    # print("BOT: How may I call you?")
    # user_name = input()
    # user_UI = UserUI(user_name)

    my_gui = SampleGUI()
    # run the event loop
    my_gui.run()
    #pub.sendMessage('msg_for_user', arg={'sentence': "How can I help you?"})
    #print("How can I help you?")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

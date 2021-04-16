import os
import random
import json
import csv
import sys
import os
import logging
import pytreebank
import xml.etree.ElementTree as ET
from utils import detokenizer


'''

In general, examples should be of the form:

{
'options': [opt_1, opt_2, ..., opt_m]
'label': l  # index of correct option
}

opt_i is an option of the form:

{
'premise': premise # the question premise (string)
'hypothesis': h # hypothesis answer (str) we calculate conditional P(hypothesis|premise)
'unc_presmise': up # the premise for calculating uncond likelihood (str)
'unc_hypothesis': uh # the hypothesis used for calculating uncond likelihood P(hypothesis) 
                     # this will often just be hypothesis but may differ slightly for format

}

'''


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def load_examples_copa(path, return_tuple = False):
    root = ET.parse(path).getroot()
    examples_copa = []
    for type_tag in root.findall('item'):
        # xml stuff
        value = type_tag.get('most-plausible-alternative')
        asks_for = type_tag.get('asks-for')
        children = type_tag.getchildren()
        # get the texts
        p = children[0].text
        a1 = children[1].text[:1].lower() +  children[1].text[1:]
        a2 = children[2].text[:1].lower() +  children[2].text[1:]
        if asks_for =='effect':
            bridge = ' so'
        elif asks_for =='cause':
            bridge = ' because'
        else: 
            assert(False)
            
        # legacy, using tuples
        if return_tuple:
            examples_copa  += [{'options': [(' '+p[:-1] ,bridge + a1),
                                                (' '+p[:-1] , bridge + a2)], 
                      'label':int(value)-1, 'asks-for': asks_for, 'bridge':bridge}]
        else:
            examples_copa  += [{'options': [{'premise':' '+p[:-1] + bridge,
                                             'hypothesis': ' '+ a1,
                                             'uncond_premise':bridge,
                                             'uncond_hypothesis':' '+a1},
                                           {'premise':' '+p[:-1] + bridge,
                                             'hypothesis': ' '+a2,
                                             'uncond_premise':bridge,
                                             'uncond_hypothesis':' '+a2}], 
                      'label':int(value)-1}]
    return examples_copa

'''

This loads COPA, putting hypothesis before the premise

(so forward LM score is PMI)

'''
## Loads from an xml
def load_examples_copa_rev(path):
    
    root = ET.parse(path).getroot()
    examples_copa = []
    for type_tag in root.findall('item'):
        # xml stuff
        value = type_tag.get('most-plausible-alternative')
        asks_for = type_tag.get('asks-for')
        children = type_tag.getchildren()
        # get the texts
        p = children[0].text[:1].lower() +  children[0].text[1:]
        a1 = children[1].text[:1].lower() +  children[1].text[1:-1]
        a2 = children[2].text[:1].lower() +  children[2].text[1:-1]
        if asks_for =='effect':
            bridge = ' because'
        elif asks_for =='cause':
            bridge = ' so'
        else: 
            assert(False)
            
        examples_copa  += [{'options': [{'premise':' '+a1 + bridge,
                                         'hypothesis':  ' ' +p,
                                         'uncond_premise':bridge,
                                         'uncond_hypothesis':' ' +p},
                                       {'premise':' '+a2 + bridge,
                                         'hypothesis': ' ' +p,
                                         'uncond_premise':bridge,
                                         'uncond_hypothesis':' '+p}], 
                            'label':int(value)-1, }]
    
    return examples_copa

def load_examples_storycloze(path, return_tuple = False):
    data = []
    with open(path) as fp:
        reader = csv.DictReader(fp, delimiter = "\t")
        for row in reader:
            d = {}
            premise = row["InputSentence1"]
            premise = f'{premise} {row["InputSentence2"]}'
            premise = f'{premise} {row["InputSentence3"]}'
            premise = f'{premise} {row["InputSentence4"]}'
            d['premise'] = premise
            hypotheses = [ row['RandomFifthSentenceQuiz1'], row['RandomFifthSentenceQuiz2'] ]
            d['hypotheses'] =  hypotheses
            correct_hypothesis = int(row['AnswerRightEnding']) - 1
            d['correct_hypothesis'] = correct_hypothesis
            d['id'] = row['InputStoryid']
            data.append(d)
    examples = []
    for d in data:
        end = '.'
        # take the punctuation from the end of the story as a prefix to 
        # the last sentence, so that we have something to condition on
        # for P(final_sentence)
        if d['premise'][-1] in '!.':
            end = d['premise'][-1] 
            d['premise'] = d['premise'][:-1]
            
        if return_tuple:
            examples += [{'options':[(d['premise'],end +' ' +h) for h in d['hypotheses']],
                        'label':d['correct_hypothesis']}]
        else:
            examples += [{'options':[{'premise':d['premise'] + end,
                                      'hypothesis':  ' ' +h,
                                      'uncond_premise': ' The story continues:' ,
                                      'uncond_hypothesis':  ' ' + h }for h in d['hypotheses']],
                        'label':d['correct_hypothesis']}]
    return examples

def load_examples_hellaswag(path):
    data = []
    with open(path) as f:
        for line in f:
            data += [json.loads(line)]
    examples = []
    for d in data:
        premise = d["ctx"].strip()
        last_space_index = premise.rfind(' ')
        uncond_premise = premise[last_space_index:]

        options = []
        for hypothesis in d['endings']:
            o = { 'premise' : premise, 'uncond_premise' : uncond_premise } 
            o['hypothesis'] = ' ' + hypothesis
            o['uncond_hypothesis'] = ' ' + hypothesis
            options.append(o)
        label = d['label']
        examples.append( { 'options' : options, 'label' : label } )
    return examples

def load_examples_cqa(path, return_tuple=False):

    examples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            label = ['A','B','C','D','E'].index(d['answerKey'])
            premise = ' ' +d['question']['stem']
            ## use the '?' as a bridge
            if not premise[-1] in '?.!':
                print(premise)
            else:
                premise = premise[:-1] ## trim the punctuation, will add a question mark
                
                
            if return_tuple:
                options = [ '? the answer is: "{}"'.format(c['text'].lower()) for c in d['question']['choices']]
                examples += [{'options': [(premise,opt) for opt in options], 
                  'label':label}]
            else:
                options = [ '? {}'.format(c['text'].lower()) for c in d['question']['choices']]
                examples += [{'options': [{'premise':premise + '? the answer is:' ,
                                          'hypothesis': ' "{}"'.format(c['text'].lower()),
                                           'uncond_premise': ' the answer is:',
                                           'uncond_hypothesis': ' "{}"'.format(c['text'].lower())} for c in d['question']['choices']], 
                          'label':label}]
    return examples

def load_examples_arc(path):
    idx2abc = { 0 : 'A', 1 : 'B', 2 : 'C', 3 : 'D', 4 : 'E' }
    abc2idx = { 'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3, 'E' : 4, '1' : 0, '2' : 1, '3' : 2, '4' : 3, '5' : 4 }
    
    examples = []
    with open(path) as lines:
        for line in lines:
            j = json.loads(line)
            d = {}

            final_label = j['answerKey']
            correct_hypothesis = abc2idx[final_label]
            q = j['question']
            stem = q['stem']
            choices = q['choices']
            hypotheses = []
            for idx, choice in enumerate(choices):
                text = choice['text']
                label = choice['label']
                assert(abc2idx[label] == idx)
                hypotheses.append(text)

            d['premise'] = stem
            d['hypotheses'] = hypotheses
            d['correct_hypothesis'] = correct_hypothesis

            d['stem'] = stem
            d['answers'] = choices
            d['label'] = final_label

            premise = d['premise']
            options = []
            for h in d['hypotheses']:
                o = {}
                h = ' ' + h
                o['premise'] = premise
                o['hypothesis'] = h
                o['uncond_premise'] = ' the answer is:'
                o['uncond_hypothesis'] = h
                options.append(o)
            label = d['correct_hypothesis']
            examples.append({'options' : options, 'label' : label })

    return examples

def load_examples_race(path, split, version):
    conversion = { 'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3, 'E' : 4, 'F' : 5 }

    examples = []


    files = [f for f in os.listdir(path=path + '{}/{}'.format(split,version)) if f.endswith('.txt')]
    for f in files:
        with open(path + '{}/{}/{}'.format(split,version, f)) as lines, open(sys.argv[2], 'w') as out:
            for line in lines:
                j = json.loads(line)

                context_id = j['id']
                context = j['article']
                ps = j['questions']
                hs = j['options']
                cs = j['answers']
                for p, h, c in zip(ps, hs, cs):
                    d = {}

                    premise = p.strip()
                    post_period = False
                    if '_' in premise:
                        idx = premise.index('_')
                        premise = premise[:idx].strip()
                        if p[-1] == '.':
                            post_period  = True
                            premise = f'{context}\n\nExplanation: {premise}'
                        elif p[-1] == '?':
                            premise = f'{context}\n\nQuestion: {premise}\n\nAnswer:'
                    elif premise[-1] == '?':
                        premise = f'{context}\n\nQuestion: {premise}\n\nAnswer:'
                    else:
                        premise = f'{context}\n\n{premise}'

                    d['premise'] = premise

                    hypotheses = [ f' {hypothesis}' for hypothesis in h ]
                    d['hypotheses'] = hypotheses

                    correct_hypothesis = conversion[c]
                    d['correct_hypothesis'] = correct_hypothesis

                    post_hypothesis = '.' if post_period else ''
                    d['post_hypothesis'] = post_hypothesis

                    d['context_id'] = context_id
                    d['context'] = context

                    d['question'] = p

                    ## this (below) is from the jsonl to examples file

                    context = d['context'].strip() 
                    question = d['question'].strip()
                    if question[0] == '.':
                        question = question[1:]
                    options = []
                    for h in d['hypotheses']:
                        o = {}
                        if '_' in question:
                            u_idx = question.find('_') 
                            premise = f' {context}\n {question[:u_idx].strip()}'
                            h = f' {h} {question[u_idx+1:].strip()}'
                            uncond_premise = '?'
                        else:
                            premise = f' {context}\n question: {question} \n answer:'
                            uncond_premise = '?'
                            h = f' {h}'
                        o['premise'] = premise
                        o['hypothesis'] = h
                        o['uncond_premise'] = uncond_premise
                        o['uncond_hypothesis'] = h
                        options.append(o)
                    label = d['correct_hypothesis']
                    examples.append({'options': options, 'label' : label })

    return examples


def load_examples_rte(path):
    data = []
    with open(path) as f:
        for line in f:
            data += [json.loads(line)]

    examples = []
    for d in data:
        premise = f" {d['premise']}\n question: {d['hypothesis']} true or false?\n answer:"
        options = []
        for h in [' true', ' false']:
            o = {}
            o['premise'] = premise
            o['hypothesis'] = h
            o['uncond_premise'] = ' true or false?\n answer:'
            o['uncond_hypothesis'] = h
            options.append(o)
        label = 0 if d['label'] == 'entailment' else 1
        examples.append({'options' : options, 'label' : label })
    return examples

def load_examples_cb(path):
    data = []
    with open(path) as f:
        for line in f:
            data += [json.loads(line)]

    examples = []
    for d in data:
        premise = f" question: Given that \"{d['premise']}\" Is \"{d['hypothesis']}\" true, false, or neither?\n answer:"
        options = []
        for h in [' true', ' false', ' neither']:
            o = {}
            o['premise'] = premise
            o['hypothesis'] = h
            o['uncond_premise'] = ' the answer is:'
            o['uncond_hypothesis'] = h
            options.append(o)
        label = ["entailment",'contradiction','neutral'].index(d['label'])
        examples.append({'options' : options, 'label' : label })
    return examples


def load_examples_snli(path):
    data = []
    with open(path) as f:
        for line in f:
            data += [json.loads(line)]

    examples = []
    for d in data:
        premise = f" question: Given that \"{d['sent1']}\" Is \"{d['sent2']}\" true, false, or neither?\n answer:"
        options = []
        for h in [' true', ' false', ' neither']:
            o = {}
            o['premise'] = premise
            o['hypothesis'] = h
            o['uncond_premise'] = ' true, false, or neither?\n answer:'
            o['uncond_hypothesis'] = h
            options.append(o)
        label = d['correct_hypothesis']
        examples.append({'options' : options, 'label' : label })
    return examples



def load_examples_sst5(path):
    data = []
    with open(path) as f:
        for line in f:
            l, s = line.strip().split('\t')
            label = int(l[-1])
            d = {}
            d['correct_hypothesis'] = label-1
            d['sentence'] = s
            data.append(d)

    examples = []
    for d in data:
        premise = f"\"{d['sentence']}\" has a tone that is"
        options = []
        for h in [' very negative.', ' somewhat negative.', ' neutral.', ' somewhat positive.', ' very positive.']:
            o = {}
            h = h + '<|endoftext|>'
            o['premise'] = premise
            o['hypothesis'] = h
            o['uncond_premise'] = ' The quote has a tone that is'
            o['uncond_hypothesis'] = h
            options.append(o)
        label = d['correct_hypothesis']
        examples.append({'options' : options, 'label' : label })
    return examples

def load_examples_sst2(path):
    data = []
    with open(path) as f:
        for line in f:
            l, s = line.strip().split('\t')
            label = int(l[-1])-3
            if label == 0:
                continue
            d = {}
            d['correct_hypothesis'] = 1 if label > 0 else 0
            d['sentence'] = s
            data.append(d)

    examples = []
    for d in data:
        premise = f"\"{d['sentence']}\" has a tone that is"
        options = []
        for h in [' negative.<|endoftext|>', ' positive.<|endoftext|>']:
            o = {}
            o['premise'] = premise
            o['hypothesis'] = h
            o['uncond_premise'] = ' The quote has a tone that is'
            o['uncond_hypothesis'] = h
            options.append(o)
        label = d['correct_hypothesis']
        examples.append({'options' : options, 'label' : label })
    return examples

def load_examples_agn(path):
    topics = [ 'World', 'Sports', 'Business', 'Science' ] 
    examples = []
    with open(path) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            label = int(row['Class Index'])-1
            title = row['Title']
            summary = row['Description']
            premise = f" title: {title}\n summary: {summary}\n topic:"
            options = []
            for h in topics:
                o = {}
                o['premise'] = premise
                o['hypothesis'] = ' ' + h.lower()
                o['uncond_premise'] = '\n topic:'
                o['uncond_hypothesis'] = ' ' + h.lower()
                options.append(o)
            label = label
            examples.append({'options' : options, 'label' : label })
    return examples


def load_examples_dbpedia(path):
    lmnames = (
        "Company",
        "Education",
        "Artist",
        "Athlete",
        "Politician",
        "Transportation",
        "Place",
        "Nature",
        "Village",
        "Species",
        "Plant",
        "Album",
        "Movie",
        "Book"
    )
    
    examples = []
    with open(path) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            d = {}
            label = int(row['Class'])-1
            lmname = lmnames[label]
            premise = f"\n excerpt: {row['Text']}\n topic:"
            options = []
            for h in lmnames:
                o = {}
                o['premise'] = premise
                o['hypothesis'] = ' ' + h
                o['uncond_premise'] = '\n topic:'
                o['uncond_hypothesis'] = ' ' + h
                options.append(o)
            examples.append({'options' : options, 'label' : label })
    return examples

def load_examples_obqa(path):
    with open(path) as lines:
        idx2abc = { 0 : 'A', 1 : 'B', 2 : 'C', 3 : 'D' }
        abc2idx = { 'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3 }

        examples = []
        for line in lines:
            j = json.loads(line)
            d = {}

            label = j['answerKey']
            correct_hypothesis = abc2idx[label]
            q = j['question']
            stem = q['stem']
            choices = q['choices']
            hypotheses = []
            for idx, choice in enumerate(choices):
                text = choice['text']
                label = choice['label']
                assert(abc2idx[label] == idx)
                hypotheses.append(text)

            d['premise'] = stem
            d['hypotheses'] = hypotheses
            d['correct_hypothesis'] = correct_hypothesis

            d['stem'] = stem
            d['answers'] = choices
            d['label'] = label





            premise = d['premise']
            options = []
            for h in d['hypotheses']:
                o = {}
                h = ' ' + h
                o['premise'] = premise
                o['hypothesis'] = h
                o['uncond_premise'] = ' the answer is:'
                o['uncond_hypothesis'] = h
                options.append(o)
            label = d['correct_hypothesis']
            examples.append({'options' : options, 'label' : label })
    return examples


def proc_passage(s):
    s = s.replace("``", '"')
    s = s.replace("''", '"')
    return s



def proc_question(s):
    s = s[0].upper() + s[1:]
    s = s.replace(' i ', ' I ')
    s = s + '?'
    return s

def load_examples_boolq(path):
    data = []
    with open(path) as f:
        for line in f:
            data += [json.loads(line)]

    examples = []
    for d in data:
        options = []
        p = f' title: { d["title"]}\n question: {proc_question(d["question"])}\n answer:'
        for h in [' yes', ' no']:
            o = {}
            o['premise'] = p
            o['hypothesis'] = h
            # o['uncond_premise'] = ' yes or no?\n answer:'
            o['uncond_premise'] = '\n answer:'
            o['uncond_hypothesis'] = h
            options.append(o)
        label = 1 if not d['answer'] else 0 #.strip().lower() == 'false' else 1
        examples.append({'options' : options, 'label' : label })
    return examples

def load_examples_sst(input_directory, curr_set="dev", granularity="binary"):
    """
    Loads the Stanford Sentiment Treebank with binary / fine-grained labels
    :param input_directory: the dataset directory
    :param curr_set: train, test, or dev
    :param granularity: binary / fine
    :return: a list of examples
    """
    # Load dataset
    dataset = pytreebank.load_sst(input_directory)

    examples = []

    for item in dataset[curr_set]:
        root = item.to_labeled_lines()[0]
        fine_grained_label, sentence = root

        # Detokenize the sentence, i.e. remove trailing spaces etc.
        premise = detokenizer(sentence.strip())

        if not premise.endswith("."):
            premise = premise + "."

        # Fine grained
        if granularity == "fine":
            opts = [" very negative.", " somewhat negative.", " neutral.", " somewhat positive.", " very positive."]
            label = fine_grained_label
        # Binary
        else:
            # Omit neutral examples from the binary version
            if fine_grained_label == 2:
                continue

            opts = [" negative.", " positive."]
            label = 0 if fine_grained_label < 2 else 1
            
            
        #premise = f"\"{d['sentence']}\" has a tone that is"
        
        
        
        premise = f"\"{premise}\" has a tone that is"
        options = []
        for h in opts:
            o = {}

            h = h + '<|endoftext|>'
            o['premise'] = premise
            o['hypothesis'] = h 
            o['uncond_premise'] = ' The quote has a tone that is'
            o['uncond_hypothesis'] = h
            options.append(o)
        label = label
        examples.append({'options' : options, 'label' : label })


    return examples

def load_examples_trec(path):
    
    label2template = [(0, 'DESC', 'a description.'),
                      (1, 'ENTY', 'an entity.'),
                      (2, 'LOC', 'a location.'),
                      (3, 'NUM', 'a number.'),
                      (4, 'ABBR', 'an abbreviation.'),
                      (5, 'HUM', 'a person.')]
    # get index of the label string

    examples = []
    
    # params
    with open(path) as f:
        for line in f:
            label = line[:line.index(' ')].split(':')[0]
            question = detokenizer(line[line.index(' ') + 1:]).strip()

            ex = {}
            options = []
            for label_idx, label_surface_form, h in label2template:
                opt = {} 
                opt['premise'] = f' {question} The answer to this question will be'
                opt['hypothesis'] = f' {h}'
                opt['uncond_premise'] = ' The answer to this question will be'
                opt['uncond_hypothesis'] = f' {h}'
                options.append(opt)
                if label_surface_form == label:
                    ex['label'] = label_idx
            ex['options'] = options
            examples.append(ex)

    return examples


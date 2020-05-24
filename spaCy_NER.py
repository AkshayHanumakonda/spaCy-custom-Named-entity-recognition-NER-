
# Code to convert text to spaCy NER format 


import re
r = re.compile('(?! )[^[]+?(?= *\[)'
               '|'
               '\[.+?\)')
sample_list = []
label_list = []
for string in intent_list:
    entity_list = []
    intent = ""
    count = 0
    print("=====>>",string)
    
    for word in r.findall(string):
        print(word)
        if word[0] == "[":
            entity_list.append((re.search(r"\(([A-Za-z0-9_]+)\)", word).group(1), re.search(r"\[.*?]", word).group(0).replace("[","").replace("]","")))
            if count != 0:
                intent = intent+" "+re.search(r"\[.*?]", word).group(0).replace("[","").replace("]","")+" "
            elif count == 0 and string[0] != "[":
                intent = intent+" "+re.search(r"\[.*?]", word).group(0).replace("[","").replace("]","")+" "
            else:
                intent = intent+re.search(r"\[.*?]", word).group(0).replace("[","").replace("]","")+" "
            count+=1
        else:
            intent = intent+word
    entity_sub_list = []
    for entities in entity_list:
        label_list.append(entities[0])
        entity_sub_list.append((intent[:-1].find(entities[1]),intent[:-1].find(entities[1])+len(entities[1]),entities[0]))
    sample_list.append((intent[:-1],{"entities":entity_sub_list}))


# Splitting data into train and test in 1:0.005 ratio

import random
from sklearn.model_selection import train_test_split
train_list ,test_list = train_test_split(sample_list,test_size=0.005) 

print(len(sample_list))
print(len(train_list))
print(len(test_list))


# Training format for spaCy NER model

train_list


# # Custom NER Training

import spacy
import plac
from pathlib import Path
import random
from tqdm import tqdm
@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def main(model=None, new_model_name='NDAP_NER', output_dir=None, n_iter=20, TRAIN_DATA = None, labels_list = None):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe('ner')
    for labels in labels_list:
        ner.add_label(labels)   # add new entity label to entity recognizer
    if model is None:
        optimizer = nlp.begin_training()
    else:
        # Note that 'begin_training' initializes the models, so it'll zero out
        # existing entity types.
        optimizer = nlp.entity.create_optimizer()

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            try:
                for text, annotations in tqdm(TRAIN_DATA):
                    nlp.update([text], [annotations], sgd=optimizer, drop=0.35,
                               losses=losses)
                print(losses)
            except:
                pass

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

main(output_dir = "./Ner_Model",TRAIN_DATA = train_list, labels_list = list(set(label_list)))


# # Custom NER Testing on test data

model_path = "./Ner_Model"
custom_NER_model = spacy.load(model_path)
count = 1
for text_string in test_list:
    string = text_string[0]
    NER_doc = custom_NER_model(string)
    print(str(count)+": "+string)
    count+=1
    for ent in NER_doc.ents:
        print(ent.label_, ent.text)
    print("\n")


# # To test a query

string= " enter a query "

NER_doc = custom_NER_model(string)
for ent in NER_doc.ents:
    print(ent.label_, ent.text)
print("\n")




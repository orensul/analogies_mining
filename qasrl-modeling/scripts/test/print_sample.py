import codecs
import json
from tqdm import tqdm
import sys

def render_question(inflected_forms, question_slots):
    # need 'pastParticiple' to be replaced before 'past' since 'past' is a substring of it...
    verb_slot = question_slots["verb"].replace("pastParticiple", inflected_forms["pastParticiple"])
    for inflection, form in inflected_forms.items():
        verb_slot = verb_slot.replace(inflection, form)
    q = question_slots
    ordered_slots = [q["wh"], q["aux"], q["subj"], verb_slot, q["obj"], q["prep"], q["obj2"]]
    present_slots = filter(lambda x: x != "_", ordered_slots)
    return ' '.join(present_slots)

with codecs.open(sys.argv[1], 'r', encoding='utf8') as f:
    for line in tqdm(f):
        sentence = json.loads(line)
        tokens = sentence["sentenceTokens"]
        print('\n' + ' '.join(tokens))
        for verb in sentence["verbs"]:
            verbIndex = verb["verbIndex"]
            verbForms = verb["verbInflectedForms"]
            verbStem = verbForms["stem"]
            print('{:d}: {:s}'.format(verbIndex, verbStem))
            for entry in verb["beam"]:
                span = entry["span"]
                spanProb = entry["spanProb"]
                answer = ' '.join(tokens[span[0] : span[1]])
                print('{:.5f}: {:s}'.format(spanProb, answer))
                for question in entry["questions"]:
                    questionProb = question["questionProb"]
                    questionString = render_question(verbForms, question["questionSlots"])
                    print('\t{:.5f}: {:s}'.format(questionProb, questionString))


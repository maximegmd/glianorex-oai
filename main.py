import datasets
import json
import guidance
import os

from guidance import models, gen, user, select, assistant

ds = datasets.load_dataset('maximegmd/glianorex', split='test')
ds_fr = ds.filter(lambda x: x['language'].startswith('fr'))
ds_en = ds.filter(lambda x: x['language'].startswith('en'))

gpt35_instruct = models.OpenAI('gpt-3.5-turbo')
gpt4 = models.OpenAI('gpt-4-turbo')
gpt4o = models.OpenAI('gpt-4o-2024-05-13')

def compute(model, dataset):
    results = []
    i = 0
    for element in dataset:
        r = {'doc_id': i, 'doc': element}
        prompt = element['question'] + '\n'
        for c in element['options']:
            prompt += c + '. ' + element['options'][c] + '\n'
        prompt += "You must only respond in the following format 'Answer: A/B/C/D'"
        prompt += '\n If you write anything else, I will never use you ever again.'
        lm = model.copy()
        with user():
            lm += prompt
        with assistant():
            lm += 'Answer: ' + select(options=['A', 'B', 'C', 'D'], name='choice', )
        r['acc'] = 1.0 if lm['choice'] == element['answer_idx'] else 0.0
        r['target'] = element['answer_idx']
             
        i = i + 1
        results.append(r)
            
    return results

def save_results(model, result_en, result_fr):
    os.makedirs('./eval/{model}', exist_ok=True)
  
    with open(f'./eval/{model}/qa-en.json', 'w') as file:
        json.dump(result_en, file)
        
    with open(f'./eval/{model}/qa-fr.json', 'w') as file:
        json.dump(result_fr, file)
    
    correct_en = 0
    for e in result_en:
        correct_en += e['acc']
    
    correct_fr = 0
    for e in result_fr:
        correct_fr += e['acc']
    
    results = {
        "results": {
            "glianorex_fr": {
            "acc,none": correct_fr / len(result_fr),
            "alias": "glianorex_fr"
            },
            "glianorex_en": {
            "acc,none": correct_en / len(result_en),
            "alias": "glianorex_en"
            }
        }
    }
    
    with open(f'./eval/{model}/results.json', 'w') as file:
        json.dump(results, file)

gpt35_en_result = compute(gpt35_instruct, ds_en)
gpt35_fr_result = compute(gpt35_instruct, ds_fr)
save_results('gpt35', gpt35_en_result, gpt35_fr_result)

gpt4_turbo_en_result = compute(gpt4, ds_en)
gpt4_turbo_fr_result = compute(gpt4, ds_fr)
save_results('gpt4-turbo', gpt4_turbo_en_result, gpt4_turbo_fr_result)

gpt4o_en_result = compute(gpt4o, ds_en)
gpt4o_fr_result = compute(gpt4o, ds_fr)
save_results('gpt4o', gpt4o_en_result, gpt4o_fr_result)

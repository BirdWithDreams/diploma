import pandas as pd

with open('../my_tests/prompts', 'r') as f:
    prompts = f.readlines()

prompts = list(map(lambda x: x.split(' ', maxsplit=1), prompts))
prompts = pd.DataFrame(prompts, columns=['number', 'prompt'])
prompts.set_index('number', inplace=True)

with open('../my_tests/whisper_output.txt', 'r') as f:
    whisper_outputs = f.readlines()

whisper_outputs = list(map(lambda x: x.split(' ', maxsplit=1), whisper_outputs))
whisper_outputs = pd.DataFrame(whisper_outputs, columns=['number', 'output'])
whisper_outputs.set_index('number', inplace=True)


result = prompts.merge(whisper_outputs, how='inner', left_index=True, right_index=True)
result.to_csv('../my_tests/whisper_output.csv')

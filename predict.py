import contextlib
import fasttext
from nltk.tokenize import wordpunct_tokenize
import os
import pandas as pd
import pymorphy2
import string
import wget 

fasttext_model_name = 'lid.176.bin'

# workaround, чтобы не выводился warning (еще не исправленный в pip-релизе баг в fasttext)
# https://github.com/facebookresearch/fastText/issues/1067
fasttext.FastText.eprint = lambda x: None


def _load_fasttext_model(models_dir='models'):
    model_path = os.path.join(models_dir, fasttext_model_name)
    if not os.path.isfile(model_path):
        if not os.path.isdir(models_dir):
            os.makedirs(models_dir)
        print('Downloading model to ', models_dir)
        wget.download('https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin', model_path)

    model = fasttext.load_model(model_path)
    return model

def _remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def _predict_once(text, morph, model, russian_percent_threshold=0.5):
    # russian_percent_threshold - минимальный процент слов из русского словаря в тексте
    russian_lines_threshold = 1
    russian_lines_count = 0
    for line in text.split('\n'):
        # fasttext умеет работать только с одной "строкой" текста (без разделителей)
        if len(line) == 0:
            continue
        labels, _ = model.predict(line)
        line_prediction = labels[0]
        if line_prediction == '__label__ru':
            # дополнительно фильтруем русские слова по словарю из OpenCorpora
            russian_words_count = 0
            words = wordpunct_tokenize(_remove_punctuation(line.lower()))
            words_count = len(words)
            if words_count > 0:
                for word in words:
                    if morph.dictionary.word_is_known(word):
                        russian_words_count += 1
                if russian_words_count / words_count >= russian_percent_threshold:                 
                    russian_lines_count += 1
                    
    binary_prediction = int(russian_lines_count >= russian_lines_threshold)
            
    return binary_prediction

def predict_once(text, russian_percent_threshold=0.5):
    morph = pymorphy2.MorphAnalyzer()
    model = _load_fasttext_model()
    binary_prediction = _predict_once(text, morph, model, russian_percent_threshold)
    print(binary_prediction)

def predict(data_dir, russian_percent_threshold=0.5):
    morph = pymorphy2.MorphAnalyzer()
    model = _load_fasttext_model()

    results = []

    filenames = sorted(os.listdir(data_dir))
    for filename in filenames:
        with open(os.path.join(data_dir, filename)) as f:
            content = f.read()
            prediction = _predict_once(content, morph, model, russian_percent_threshold)
        
        results.append({
            'filename': filename,
            'answer': prediction,
        })
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(data_dir, 'prediction.csv'), index=None)
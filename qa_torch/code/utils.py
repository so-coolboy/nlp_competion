from config import*


class NLPTransform(BasicTransform):
    """ Transform for nlp task."""
    LANGS = {
        'en': 'english',
        'it': 'italian', 
        'fr': 'french', 
        'es': 'spanish',
        'tr': 'turkish', 
        'ru': 'russian',
        'pt': 'portuguese'
    }

    @property
    def targets(self):
        return {"data": self.apply}
    
    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        return params

    def get_sentences(self, text, lang='en'):
        return sent_tokenize(text, self.LANGS.get(lang, 'english'))



class ReverseSentencesTransform(NLPTransform):
    """ Do shuffle by sentence """
    def __init__(self, always_apply=False, p=0.5):
        super(ReverseSentencesTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, selected_text, lang = data
        text_copy = text
        selected_text_copy = selected_text
#         text = text.translate(str.maketrans('', '', string.punctuation))
#         selected_text = selected_text.translate(str.maketrans('', '', string.punctuation))
        text =text.split(' ')
        text = text[::-1]
        selected_text =selected_text.split(' ')
        selected_text = selected_text[::-1]
        
        text = ' '.join(text)
        selected_text = ' '.join(selected_text)
        if text.find(selected_text)==-1 or len(selected_text)==0:
            text = text_copy
            selected_text = selected_text_copy
        return text, selected_text, lang


class ExcludeNumbersTransform(NLPTransform):
    """ exclude any numbers """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeNumbersTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, selected_text, lang = data
        text_copy = text
        selected_text_copy = selected_text
        text = re.sub(r'[0-9]', '', text)
        text = re.sub(r'\s+', ' ', text)
        selected_text = re.sub(r'[0-9]', '', selected_text)
        selected_text = re.sub(r'\s+', ' ', selected_text)        
        if text.find(selected_text)==-1 or len(selected_text)==0:
            text = text_copy
            selected_text = selected_text_copy
        return text, selected_text, lang


class ExcludeHashtagsTransform(NLPTransform):
    """ Exclude any hashtags with # """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeHashtagsTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text,selected_text, lang = data
        text_copy = text
        selected_text_copy = selected_text
        text = re.sub(r'#[\S]+\b', '', text)
        text = re.sub(r'\s+', ' ', text)
        selected_text = re.sub(r'#[\S]+\b', '', selected_text)
        selected_text = re.sub(r'\s+', ' ', selected_text)
        if text.find(selected_text)==-1 or len(selected_text)==0:
            text = text_copy
            selected_text = selected_text_copy
        return text, selected_text, lang


class ExcludeUsersMentionedTransform(NLPTransform):
    """ Exclude @users """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeUsersMentionedTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text,selected_text, lang = data
        text_copy = text
        selected_text_copy = selected_text
        text = re.sub(r'@[\S]+\b', '', text)
        text = re.sub(r'\s+', ' ', text)
        selected_text = re.sub(r'@[\S]+\b', '', selected_text)
        selected_text = re.sub(r'\s+', ' ', selected_text)
        
        if text.find(selected_text)==-1 or len(selected_text)==0:
            text = text_copy
            selected_text = selected_text_copy
        return text, selected_text, lang


class ExcludeUrlsTransform(NLPTransform):
    """ Exclude urls """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeUrlsTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text,selected_text, lang = data
        text_copy = text
        selected_text_copy = selected_text
        text = re.sub(r'https?\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        selected_text = re.sub(r'https?\S+', '', selected_text)
        selected_text = re.sub(r'\s+', ' ', selected_text)
        
        if text.find(selected_text)==-1 or len(selected_text)==0:
            text = text_copy
            selected_text = selected_text_copy
        return text, selected_text, lang


class ExcludeDuplicateSentencesTransform(NLPTransform):
    """ Exclude equal sentences """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeDuplicateSentencesTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text,selected_text, lang = data
        text_copy = text
        selected_text_copy = selected_text
        text_s = []
        for sentence in self.get_sentences(text, lang):
            sentence = sentence.strip()
            if sentence not in text_s:
                text_s.append(sentence)
        selected_text_s = []
        for sentence in self.get_sentences(selected_text, lang):
            sentence = sentence.strip()
            if sentence not in selected_text_s:
                selected_text_s.append(sentence)
        text = ' '.join(text_s)
        selected_text = ' '.join(selected_text_s)
        
        if text.find(selected_text)==-1 or len(selected_text)==0:
            text = text_copy
            selected_text = selected_text_copy
        return text, selected_text, lang

if __name__ == '__main__':

    transform = ReverseSentencesTransform(p=1.0)

    text = '<Sentence1>. <Sentence2>. <Sentence3>. <Sentence4>. <Sentence5>. <Sentence6>.'
    selected_text = '<Sentence2>. <Sentence3>. <Sentence4>.'
    lang = 'en'

    print(transform(data=(text, selected_text,lang))['data'][0])
    print(transform(data=(text, selected_text,lang))['data'][1])


    transform = ExcludeNumbersTransform(p=1.0)

    text = '<Word1> <Word2> <Word3> <Word4> <Word5> <Word6> <Word7> <Word8> <Word9> <Word10>'
    selected_text = '<Word3> <Word4> <Word5> <Word6>'
    lang = 'en'

    print(transform(data=(text, selected_text,lang))['data'][0])
    print(transform(data=(text, selected_text,lang))['data'][1])


    transform = ExcludeHashtagsTransform(p=1.0)

    text = '<Word1> <Word2> <Word3> #kaggle <Word4> <Word5> <Word6> <Word7> <Word8> <Word9> <Word10>'
    selected_text = '<Word3> #kaggle <Word4> <Word5> <Word6>'
    lang = 'en'

    print(transform(data=(text, selected_text,lang))['data'][0])
    print(transform(data=(text, selected_text,lang))['data'][1])


    transform = ExcludeUsersMentionedTransform(p=1.0)

    text = '<Word1> <Word2> <Word3> @kaggle <Word4> <Word5> <Word6> <Word7> <Word8> <Word9> <Word10>'
    selected_text = ' @kaggle <Word4> <Word5>'
    lang = 'en'

    print(transform(data=(text, selected_text,lang))['data'][0])
    print(transform(data=(text, selected_text,lang))['data'][1])


    transform = ExcludeUrlsTransform(p=1.0)

    text = '<Word1> <Word2> <Word3> <Word4> https://www.kaggle.com/shonenkov/nlp-albumentations/ <Word6> <Word7> <Word8> <Word9> <Word10>'
    selected_text = '<Word4> https://www.kaggle.com/shonenkov/nlp-albumentations/ <Word6> <Word7>'
    lang = 'en'

    print(transform(data=(text, selected_text,lang))['data'][0])
    print(transform(data=(text, selected_text,lang))['data'][1])


    transform = ExcludeDuplicateSentencesTransform(p=1.0)

    text = '<Sentence1>. <Sentence2>. <Sentence4>. <Sentence4>. <Sentence5>. <Sentence5>.'
    selected_text = '<Sentence4>. <Sentence4>. <Sentence5>.'
    lang = 'en'

    print(transform(data=(text, selected_text,lang))['data'][0])
    print(transform(data=(text, selected_text,lang))['data'][1])








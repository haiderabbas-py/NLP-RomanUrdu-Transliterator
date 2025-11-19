import re
import unicodedata
DIACRITICS_RE = re.compile('[\u0610-\u061A\u06D6-\u06ED]')
def normalize_urdu(text):
    text=unicodedata.normalize('NFC',text)
    text=DIACRITICS_RE.sub('',text)
    text=text.replace('\u0622','\u0627').replace('\u06CC','\u064A')
    return re.sub('\s+',' ',text).strip()
_SIMPLE_MAP={'ا':'a','ب':'b','پ':'p','ت':'t','ٹ':'t','ث':'s','ج':'j','چ':'ch','ح':'h','خ':'kh','د':'d','ڈ':'d','ر':'r','ڑ':'r','ز':'z','ژ':'zh','س':'s','ش':'sh','ص':'s','ض':'z','ط':'t','ظ':'z','ع':''','غ':'gh','ف':'f','ق':'q','ک':'k','گ':'g','ل':'l','م':'m','ن':'n','و':'v','ہ':'h','ء':''','ی':'y','ے':'e'}
def simple_romanize(text):
    text=normalize_urdu(text)
    return ''.join(_SIMPLE_MAP.get(c,c) for c in text)

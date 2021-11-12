# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import re
import string
import zipfile
import sys
sys.path.append('E:/Code/Python/sumy/sumy/nlp/')
import nltk

from .._compat import to_string, to_unicode, unicode
from ..utils import normalize_language


class DefaultWordTokenizer(object):
    """NLTK tokenizer"""
    def tokenize(self, text):
        return nltk.word_tokenize(text)


class HebrewWordTokenizer:
    """https://github.com/iddoberger/awesome-hebrew-nlp"""
    _TRANSLATOR = str.maketrans("", "", string.punctuation)

    @classmethod
    def tokenize(cls, text):
        try:
            from hebrew_tokenizer import tokenize
            from hebrew_tokenizer.groups import Groups
        except ImportError:
            raise ValueError("Hebrew tokenizer requires hebrew_tokenizer. Please, install it by command 'pip install hebrew_tokenizer'.")

        text = text.translate(cls._TRANSLATOR)
        return [
            word for token, word, _, _ in tokenize(text)
            if token in (Groups.HEBREW, Groups.HEBREW_1, Groups.HEBREW_2)
        ]


class JapaneseWordTokenizer:
    def tokenize(self, text):
        try:
            import tinysegmenter
        except ImportError as e:
            raise ValueError("Japanese tokenizer requires tinysegmenter. Please, install it by command 'pip install tinysegmenter'.")
        segmenter = tinysegmenter.TinySegmenter()
        return segmenter.tokenize(text)


class ChineseWordTokenizer:
    def tokenize(self, text):
        try:
            import jieba
        except ImportError as e:
            raise ValueError("Chinese tokenizer requires jieba. Please, install it by command 'pip install jieba'.")
        return jieba.cut(text)


class KoreanSentencesTokenizer:
    def tokenize(self, text):
        try:
            from konlpy.tag import Kkma
        except ImportError as e:
            raise ValueError("Korean tokenizer requires konlpy. Please, install it by command 'pip install konlpy'.")
        kkma = Kkma()
        return kkma.sentences(text)


class KoreanWordTokenizer:
    def tokenize(self, text):
        try:
            from konlpy.tag import Kkma
        except ImportError as e:
            raise ValueError("Korean tokenizer requires konlpy. Please, install it by command 'pip install konlpy'.")
        kkma = Kkma()
        return kkma.nouns(text)


class VietNameseSentencesTokenizer:
    def tokenize(self, text):
        try:
            from vncorenlp import VnCoreNLP
        except ImportError as e:
            raise ValueError("VietNamese requires vncorenlp. Please, install it by command 'pip install vncorenlp'.")
        rdrsegmenter = VnCoreNLP("E:/Code/Python/sumy/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
        sentences = rdrsegmenter.tokenize(text) 
        sentences_tokened = []
        for sentence in sentences:
            sentences_tokened.append(" ".join(sentence).strip())
        return sentences_tokened
        

class VietNameseWordsTokenizer:
    def tokenize(self, text):
        return [w.strip() for w in text.split(" ")]    


class Tokenizer(object):
    """Language dependent tokenizer of text document."""

    _WORD_PATTERN = re.compile(r"^[^\W\d_](?:[^\W\d_]|['-])*$", re.UNICODE)
    # feel free to contribute if you have better tokenizer for any of these languages :)
    LANGUAGE_ALIASES = {
        "slovak": "czech",
    }

    # improve tokenizer by adding specific abbreviations it has issues with
    # note the final point in these items must not be included
    LANGUAGE_EXTRA_ABREVS = {
        "english": ["e.g", "al", "i.e"],
        "german": ["al", "z.B", "Inc", "engl", "z. B", "vgl", "lat", "bzw", "S"],
    }

    SPECIAL_SENTENCE_TOKENIZERS = {
        'hebrew': nltk.RegexpTokenizer(r'\.\s+', gaps=True),
        'japanese': nltk.RegexpTokenizer('[^　！？。]*[！？。]'),
        'chinese': nltk.RegexpTokenizer('[^　！？。]*[！？。]'),
        'korean': KoreanSentencesTokenizer(),
        'vietnamese': VietNameseSentencesTokenizer(),
    }

    SPECIAL_WORD_TOKENIZERS = {
        'hebrew': HebrewWordTokenizer(),
        'japanese': JapaneseWordTokenizer(),
        'chinese': ChineseWordTokenizer(),
        'korean': KoreanWordTokenizer(),
        'vietnamese': VietNameseWordsTokenizer(),
    }

    def __init__(self, language):
        language = normalize_language(language)
        self._language = language

        tokenizer_language = self.LANGUAGE_ALIASES.get(language, language)
        self._sentence_tokenizer = self._get_sentence_tokenizer(tokenizer_language)
        self._word_tokenizer = self._get_word_tokenizer(tokenizer_language)

    @property
    def language(self):
        return self._language

    def _get_sentence_tokenizer(self, language):
        if language in self.SPECIAL_SENTENCE_TOKENIZERS:
            return self.SPECIAL_SENTENCE_TOKENIZERS[language]
        try:
            path = to_string("tokenizers/punkt/%s.pickle") % to_string(language)
            return nltk.data.load(path)
        except (LookupError, zipfile.BadZipfile) as e:
            raise LookupError(
                "NLTK tokenizers are missing or the language is not supported.\n"
                """Download them by following command: python -c "import nltk; nltk.download('punkt')"\n"""
                "Original error was:\n" + str(e)
            )

    def _get_word_tokenizer(self, language):
        if language in self.SPECIAL_WORD_TOKENIZERS:
            return self.SPECIAL_WORD_TOKENIZERS[language]
        else:
            return DefaultWordTokenizer()

    def to_sentences(self, paragraph):
        if hasattr(self._sentence_tokenizer, '_params'):
            extra_abbreviations = self.LANGUAGE_EXTRA_ABREVS.get(self._language, [])
            self._sentence_tokenizer._params.abbrev_types.update(extra_abbreviations)
        sentences = self._sentence_tokenizer.tokenize(to_unicode(paragraph))
        return tuple(map(unicode.strip, sentences))

    def to_words(self, sentence):
        words = self._word_tokenizer.tokenize(to_unicode(sentence))
        return tuple(filter(self._is_word, words))

    @staticmethod
    def _is_word(word):
        return bool(Tokenizer._WORD_PATTERN.match(word))


if __name__ == "__main__":
    text = "Tổng thư ký Tổ chức Hiệp ước Bắc Đại Tây Dương (NATO) Jens Stoltenberg ngày 19/5 cho biết, nếu Ai Cập đề nghị, liên minh này sẽ hỗ trợ công tác tìm kiếm chiếc máy bay mang số hiệu MS 804 của hãng hàng không Ai Cập chở 66 người mất tích trước đó cùng ngày.\n“Tôi gửi lời chia buồn sâu sắc nhất đến những ai bị ảnh hưởng bởi vụ việc này. Tôi cũng gửi lời chia buồn sâu sắc đến Pháp và Ai Cập. Tôi biết rằng đã có những nỗ lực tìm kiếm cứu nạn ở mức độ quốc gia. Pháp và Ai Cập đang phối hợp trong công tác này cũng như việc điều tra. Chúng tôi sẽ tiếp tục theo dõi chặt chẽ diễn biến và nếu được đề nghị, NATO luôn sẵn sàng giúp đỡ”, ông Jens Stoltenberg nói.\nThủ tướng Italy Matteo Renzi ngày 19/5 cũng đã gửi lời chia buồn, đồng thời bày tỏ sự đoàn kết với Ai Cập sau vụ máy bay của hãng hàng không Ai Cập mất tích trên Địa Trung Hải khi đang trên đường bay từ Paris đến Cairo.\nTrước đó, Hãng hàng không quốc gia Ai Cập (EgyptAir) xác nhận phía Hy Lạp đã tìm thấy mảnh vỡ từ chiếc máy bay này ở phía Nam đảo Karpathos, thuộc vùng Nam Địa Trung Hải. Hãng đã gửi lời chia buồn đến gia đình các hành khách trên chuyến bay mất tích như một sự xác nhận đầu tiên rằng thân nhân của họ đã qua đời. Hãng cũng cam kết sẽ triển khai mọi biện pháp giải quyết tình hình hiện nay cũng như tiến hành một cuộc điều tra tổng thể.\nNgười đứng đầu cơ quan điều tra tai nạn hàng không Ai Cập Ayman al-Moqadem ngày 19/5 cho biết, nước này sẽ dẫn đầu một ủy ban điều tra về vụ mất tích chiếc máy bay mang số hiệu MS 804.\nỦy ban này bao gồm cả nhân sự phía Pháp, nước sản xuất chiếc Airbus 320 này và cũng là nước có số nạn nhân nhiều thứ hai sau Ai Cập. Cơ quan chức năng Pháp đã khẳng định sẽ cử 3 chuyên gia sang Ai Cập tham gia điều tra vụ tai nạn máy bay này. Anh và Hy Lạp cũng đã đề nghị giúp đỡ tìm kiếm hộp đen và những mảnh vỡ của chiếc máy bay.\nHội đồng an toàn giao thông quốc gia Mỹ cho biết, động cơ của chiếc máy bay gặp nạn được sản xuất tại nước này. Theo quy tắc quốc tế, nước nơi động cơ máy bay được chế tạo cũng có thể tham gia vào cuộc điều tra khi tai nạn xảy ra. Hiện Mỹ đã cử máy bay P-3 Orion hỗ trợ công tác tìm kiếm chiếc máy bay mất tích của Ai Cập.\nLúc này, ứng viên Tổng thống đảng Cộng hòa Mỹ Donald Trump đã lên tiếng bày tỏ nghi ngờ đây là một vụ tấn công khủng bố song chính phủ Mỹ cho rằng, vụ tai nạn máy bay vẫn đang được điều tra và còn quá sớm để xác định nguyên nhân khiến máy bay gặp nạn.\nThủ tướng Ai Cập Sherif Ismail thì nhận định còn quá sớm đề loại bỏ bất cứ giả thuyết nào, kể cả trường hợp máy bay bị khủng bố. Bộ trưởng Bộ Hàng không dân dụng Ai Cập Sherif Phathi cũng cho rằng, khả năng máy bay bị khủng bố cao hơn khả năng xảy ra lỗi kỹ thuật dù ông chưa đưa ra bằng chứng cụ thể nào.\nTổng thống Ai Cập Mohamed Morsi đã yêu cầu Bộ Hàng không dân dụng và quân đội phối hợp nhanh chóng định vị nơi chiếc máy bay mang số hiệu MS 804 rơi và tiến hành một cuộc điều tra thấu đáo.\nTrong khi đó, Ngoại trưởng Canada Stephane Dion ngày 19/5 cho biết trong số những hành khách đi chuyến bay mang số hiệu MS 804 của Hãng hàng không quốc gia Ai Cập (EgyptAir) bị mất tích cùng ngày có ít nhất 2 công dân nước này. Ông cũng cho biết Bộ Ngoại giao Canada đang phối hợp với các đối tác Pháp và Ai Cập, cũng như các nước liên quan khác để đánh giá tình hình và xem xét các yêu cầu hỗ trợ.\nTrước đó, hãng hàng không quốc gia Ai Cập đã công bố quốc tịch của những hành khách đi trên chuyến bay MS 804 bị mất tích, bao gồm 56 hành khách, trong đó có 30 người Ai Cập, 15 người Pháp, 2 người I-rắc, 1 người Anh, 1 người Bỉ, 1 người Kuwait, 1 người Saudi Arabia, 1 người Sudan, 1 người Cộng hòa Chad, 1 người Bồ Đào Nha, 1 người Angeria và 1 người Canada. Ngoài ra, còn có 10 thành viên phi hành đoàn./.\n"
    sentences = VietNameseSentencesTokenizer().tokenize(text)
    words = VietNameseWordsTokenizer().tokenize(sentences[0])
    print(sentences[0])
    print(words)
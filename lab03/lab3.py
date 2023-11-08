import nltk
import gensim
from gensim import corpora
import re
import string

from gensim.corpora import dictionary
from sklearn.feature_extraction.text import TfidfVectorizer

# Завантаження стоп-слів з NLTK
nltk.download('stopwords')
from nltk.corpus import stopwords

def text_preprocessing(text):
    # Крок 1: Приведення тексту до нижнього регістру
    text = text.lower()

    # Крок 2: Видалення цифр та неалфавітних символів
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Крок 3: Токенізація тексту
    words = text.split()

    # Крок 4: Видалення стоп-слів
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Крок 5: Видалення знаків пунктуації
    words = [word.strip(string.punctuation) for word in words]

    # Повертаємо оброблений текст у вигляді рядка
    return ' '.join(words)



# Визначення топ-20 слів в кожній главі
top_words_per_chapter_TF_IDF = []
top_words_per_chapter_LDA = []

def split_text_into_chapters(text):
    chapters = text.split("chapter")
    chapters[0] = "chapter" + chapters[0]
    chapters = chapters[13:]
    return chapters

def get_top_words_by_TF_IDF(chapters):
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), max_features=20)


    for chapter_text in chapters:
        # Обробка тексту (аналогічно, як у попередній відповіді)
        chapter_text = chapter_text.lower()
        chapter_text = re.sub(r'[^a-zA-Z\s]', '', chapter_text)
        words = chapter_text.split()
        words = [word.strip(string.punctuation) for word in words]

        # Застосування TF-IDF до глави
        tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(words)])
        feature_names = tfidf_vectorizer.get_feature_names_out()

        # Вибір топ-20 слів за значенням TF-IDF
        top_words = [feature_names[i] for i in tfidf_matrix.indices]
        top_words_per_chapter_TF_IDF.append(top_words)

def count_common_words(string1, string2):
    words_1 = string1.split(", ")
    words_2 = string2.split(", ")
    set1 = set(words_1)
    set2 = set(words_2)

    common_words = set1.intersection(set2)
    count = len(common_words)
    return count


with open('AliceInWonderland.txt', 'r', encoding='utf-8') as file:
    # Read the entire content of the file into a string
    textAlice = file.read()


preprocessed_text = text_preprocessing(textAlice)

chapters_text = split_text_into_chapters(preprocessed_text)

token_chapters = []
for chapter in chapters_text:
    tokens = chapter.split()
    token_chapters.append(tokens)

# Побудова словників та корпусу для моделі LDA
dictionary = corpora.Dictionary(token_chapters)
corpus = [dictionary.doc2bow(tokens) for tokens in token_chapters]

# Побудова моделі LDA
lda_model = gensim.models.LdaModel(corpus, num_topics=12, id2word=dictionary)

get_top_words_by_TF_IDF(chapters_text)

for i, chapter in enumerate(chapters_text):
    tf_idf_words = ", ".join(top_words_per_chapter_TF_IDF[i])
    lda_words = ", ".join([dictionary[word_id] for word_id, _ in lda_model.get_topic_terms(i, topn=20)])
    print("Chapter: " + str(i+1))
    print("TF_IDF words: " + tf_idf_words)
    print("LDA words: " + lda_words)

    print("Common words: " + str(count_common_words(tf_idf_words, lda_words)) + "\n")







"""
Daniel Xu
DS 3500 - Professor Rachlin
Homework #2 - NLP Library Framework
"""

import re
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from collections import Counter
from nltk.corpus import stopwords
from plotly.subplots import make_subplots


class Textastic:
    def __init__(self):
        self.word_data = defaultdict(dict)

    def _save_results(self, label, results):
        for k, v in results.items():
            self.word_data[label][k] = v

    def load_text(self, filename, label=None, parser=None, stop_words=set(stopwords.words('english'))):
        """ Load in text files, parse and clean them, and add a count of all their words to the object's
        word_data dictionary
        :param filename: name of file that is loading in
        :param label: label to be used for key in word_data dictionary
        :param parser: parser function to be used if not a .txt file
        :param stop_words: list of stopwords to clean out from text, default is NLTK module's stopwords
        :return: count of words in text added to object's word_data dictionary
        """
        if parser is None:
            script = (Textastic._default_parser(filename))
            words = Textastic.clean_words(stop_words, script)
        else:
            script = parser(filename)
            words = Textastic.clean_words(stop_words, script)

        if label is None:
            label = filename

        self._save_results(label, Textastic.get_results(words))

    @staticmethod
    def _default_parser(filename):
        """ Parses through txt file (default) and collects word counts
        :param filename: a .txt file
        :return: Count of words
        """
        text = Path(filename).read_text()
        text = text.replace('\n', '')

        return text

    @staticmethod
    def load_stop_words(stop_words, text):
        """ Load in list of stop words to filter out of strings
        :param stop_words: List of stopwords to remove
        :param text: text to filter through
        :return: Cleaned strings without stop words
        """

        # Need to do two things, first filter out stopwords and also make sure that the words are actual words
        # Problem is when I filtered out apostrophes, it created weird leftover contractions, ie. we'll --> [we, ll]
        word_lst = [word for word in text if word not in stop_words and len(word) > 2]
        return word_lst

    @staticmethod
    def clean_words(stop_words, text):
        """ Clean out parts of script that need to be removed like character names, captions, stopwords, etc..
        :param stop_words: List of stopwords to remove
        :param text: text to filter through
        :return: Cleaned script with just the dialogue
        """
        # Remove completely capitalized words, things like movement cues, character names)
        cleaned_text = re.sub(r'\b[A-Z]+\b', '', text)

        # Remove all punctuation except apostrophes
        cleaned_text = re.sub("[^\w\s]", ' ', cleaned_text)

        # Split text into list of lower cased words
        words = cleaned_text.split()
        words = [word.strip().lower() for word in words]

        return Textastic.load_stop_words(stop_words, words)

    @staticmethod
    def get_results(word_lst):
        """ Count number of words in word list and count of each word
        :param word_lst: list of words
        :return: dictionary of word counts, num words, and average word length
        """

        results = {
            'numwords': len(word_lst),
            'wordcount': Counter(word_lst),
            'word_length': round(sum(map(len, word_lst)) / float(len(word_lst)), 3)
        }
        return results

    @staticmethod
    def get_top_words(word_data, k):
        """ Return the top k words from each text in word_date
        :param k: k number of words to return
        :return: the top k words from each text and the number of occurrences
        """
        top_words = []
        source = []
        # Iterate through word_data, get k most common words out of each text, reference them to a text, and
        # add them to a dataframe
        for key, value in word_data.items():
            top_words += value['wordcount'].most_common(k)
            source += [key] * k

        df = pd.DataFrame(top_words, columns=['Word', 'Occurrences'])
        df.insert(0, 'Text', source)
        return df

    @staticmethod
    def code_mapping(df):
        """ Map labels in src and targ columns to integers
        :param df: a dataframe
        :return:
        """
        # Sort source and target columns, then add together
        labels = sorted(list(df['Text'])) + sorted(list(df['Word']))

        # Remove duplicates
        labels = list(set(labels))

        # Give labels corresponding integer
        codes = list(range(len(labels)))
        lcmap = dict(zip(labels, codes))

        df = df.replace({'Text': lcmap, 'Word': lcmap})

        return df, labels

    @staticmethod
    def sankey_visualization(sankey_df, labels, node):
        link = {'source': sankey_df['Text'], 'target': sankey_df['Word'], 'value': sankey_df['Occurrences']}

        if not node:
            node = {'pad': 100, 'thickness': 10, 'line': {'color': 'black', 'width': 2}, 'label': labels}
        else:
            node = node

        sk = go.Sankey(link=link, node=node)
        fig = go.Figure(sk)
        fig.show()

    def make_sankey(self, k=10, node=None):
        """ Create a sankey visualization of most common words in each text
        :param node: node features for sankey diagram
        :param k: Number of k-most common words to plot on sankey diagram
        :return: a sankey diagram
        """

        # Get k top words
        sankey_map = Textastic.get_top_words(self.word_data, k)

        # Make code mapping and sankey visualization
        sankey_df, labels = Textastic.code_mapping(sankey_map)
        Textastic.sankey_visualization(sankey_df, labels, node=node)

    def bubble_chart(self, k=10):
        """ Create bubble chart visualization of texts using plotly
        Note: I know this isn't a very good chart that represents my word data very well, but I just wanted to make
        some type of plotly graph
        :param: k number of top words
        :param: dimension of subplots
        :return: None (subplots of bubble charts)
        """
        # Get top k words and also calculate word length of each word in df
        bubble_df = Textastic.get_top_words(self.word_data, k)
        bubble_df['Word Length'] = bubble_df['Word'].str.len()

        # Create bubble chart
        fig = px.scatter(data_frame=bubble_df,
                         x='Word Length',
                         y='Occurrences',
                         color='Text',
                         hover_name='Word',
                         size='Occurrences')
        fig.show()

    def histogram(self, dimensions):
        """ Create histogram for each of the texts
        :return: figure depicting each text's words, one subplot per text
        x-axis: word length
        y-axis: occurrences
        """
        num_texts = len(self.word_data)
        assert dimensions[0] * dimensions[1] == num_texts, 'Dimensions provided does not match number of texts inputted'

        text_word_lengths = []

        # Iterate through word data, sum up # of instances of each word length
        for text, data in self.word_data.items():
            words_by_length = defaultdict(int)
            for word, occurrences in data['wordcount'].items():
                words_by_length[word] = len(word)

            # Convert dict to a dataframe
            df = pd.DataFrame.from_dict(words_by_length, orient='index', columns=['Word Length'])
            df.sort_index(inplace=True)
            df.reset_index(inplace=True)
            text_word_lengths.append(df)

        # Create subplots that match the specified dimensions
        fig, axes = plt.subplots(dimensions[0], dimensions[1])
        titles = list(self.word_data.keys())

        text_counter = 0
        for i in range(dimensions[0]):
            for j in range(dimensions[1]):
                bins = max(text_word_lengths[text_counter]['Word Length']) - 1
                sns.histplot(data=text_word_lengths[text_counter], x='Word Length', bins=bins, ax=axes[i, j])
                axes[i, j].set_title(titles[text_counter])
                text_counter += 1

        plt.tight_layout()
        fig.show()

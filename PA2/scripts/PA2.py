#!/usr/bin/env python3

import argparse
import json
import logging
from math import trunc
import math
import sys
import ebooklib
from ebooklib import epub 
from bs4 import BeautifulSoup
from collections import Counter
from random import random, seed
import matplotlib.pyplot as plt
import statistics

# create logger
logger = logging.getLogger("PA2")
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

def readEPub(name):
    """Reads an EPub from a path and transforms it into a list of strings

    Args:
        name (string): Name of the desired .epub

    Returns:
        text (list): List of strings, each element is a different book chapter
    """
    # Initialize variables
    chapters = []    
    text = [] 
    path = 'book/' + name
    # Retrieve every item in the epub
    book = epub.read_epub(path)
    # Remove every non-text element
    for item in book.get_items():        
        if item.get_type() == ebooklib.ITEM_DOCUMENT:            
            chapters.append(item.get_content())   
    # Convert from html to strings
    for chapter in chapters:
        soup = BeautifulSoup(chapter, 'html.parser')   
        soup.prettify(formatter=lambda s: s.replace('\n', ' '))
        for data in soup(['style', 'script']):
            data.decompose()
        chapter_text = ' '.join(soup.stripped_strings)
        chapter_text = chapter_text.replace('\n', ' ').replace('\r', ' ').replace('\xa0',' ')
        text.append(chapter_text)
    return text


def txtToList(path):
    """Converts a txt file to list of strings

    Args:
        path (string): path of the location of the txt file

    Returns:
        list: list of every word inside the txt file
    """
    # Opening the file in read mode
    txt_file = open(path, "r")
    # Reading the file
    data = txt_file.read()
    # Replacing end splitting the text when newline ('\n') is seen.
    data_list = data.split("\n")
    txt_file.close()

    return data_list


def totalCounter(text, elements_to_remove, stop_words_path):
    """Count the number of occurrences of each word in a list of strings

    Args:
        text (list): list of strings to be evaluated
        elements_to_remove (string): string composed of every element to be removed (such as punctuation)
        stop_words_path (string): string path of a txt file which contains common stop words to not be considered

    Returns:
        counter: counter with every occurrence of each word inside the variable text
    """

    # Initialize the counter
    counts = Counter()

    # Create a list of every stop word
    stop_words_list = txtToList(stop_words_path)

    # For every chapter, count the words not in the list of stop words
    for chapter in text:
        counts.update(word.strip(elements_to_remove).lower() for word in chapter.split() if word.strip(elements_to_remove).lower() not in stop_words_list)
    
    return counts


def fixedProbabilisticCounter(text, elements_to_remove, stop_words_path, probability):
    """Count the number of occurrences of each word in a list of strings using a fixed probabilistic counter

    Args:
        text (list): list of strings to be evaluated
        elements_to_remove (string): string composed of every element to be removed (such as punctuation)
        stop_words_path (string): string path of a txt file which contains common stop words to not be considered
        probability (int): fixed probability used for the counter

    Returns:
        counter: counter with every occurrence of each word inside the variable text
    """

    # Initialize the counter
    counts = Counter()

    # Create a list of every stop word
    stop_words_list = txtToList(stop_words_path)

    # For every chapter, count the words not in the list of stop words if a random number is lower than a certain probability
    for chapter in text:
        counts.update(word.strip(elements_to_remove).lower() for word in chapter.split() if word.strip(elements_to_remove).lower() not in stop_words_list and random() < probability)
    
    return counts


def csurosCounter(text, elements_to_remove, stop_words_path, csuros_dict):
    """Count the number of occurrences of each word in a list of strings using a csuros counter

    Args:
        text (list): list of strings to be evaluated
        elements_to_remove (string): string composed of every element to be removed (such as punctuation)
        stop_words_path (string): string path of a txt file which contains common stop words to not be considered
        csuros_dict (dict): dictionary in the format : {'d': int, 'p': int}

    Returns:
        counter: counter with every occurrence of each word inside the variable text
    """

    # Initialize the counter
    counts = Counter()

    # Create a list of every stop word
    stop_words_list = txtToList(stop_words_path)

    # For every chapter, count the words not in the list of stop words according to a csuros counter
    for chapter in text:
        counts.update(word.strip(elements_to_remove).lower() for word in chapter.split() if word.strip(elements_to_remove).lower() not in stop_words_list and random() < csuros_dict['p'] ** trunc(counts[word.strip(elements_to_remove).lower()]/(2**csuros_dict['d'])))
    
    return counts


def createStats(counter_dict, real_counter, top_word_number, book_name):
    """Create various stats for the counters

    Args:
        counter_dict (dict): Dictionary of probabilistic counter
        real_counter (counter): Counter of all words
        top_word_number (int): Number of top words to calculate stats
        book_name (str): Name of book counters were taken

    Returns:
        dict: counter_dict updated with stats
    """
    # Create variables
    top_words = dict()
    stats = {key: {'occurrence' : []} for idx, (key, occurrence) in enumerate(real_counter.most_common()) if idx < top_word_number}

    # Find real top words
    real_top_words = [(word, occurrence) for idx, (word, occurrence) in enumerate(real_counter.most_common()) if idx < top_word_number]
    real_word_list = [tup[0] for tup in real_top_words]

    # Find occurrences of top_words in counter
    for repetition, counter in counter_dict['counter'].items():
        top_words[repetition] = [(word, occurrence) for (word, occurrence) in counter.most_common() if word in real_word_list]

    # Append the occurrences to a list
    for repetition, word_list in top_words.items():
        for (key, occurrence) in word_list:
            stats[key]['occurrence'].append(occurrence)

    # For every word, calculate stats
    for key in stats.keys():
        stats[key]['real'] = real_counter[key] 
        if counter_dict['type'] == 'fixed':
            stats[key]['expected'] = stats[key]['real'] * counter_dict['parameters']['p']
        elif counter_dict['type'] == 'csuros':
            x = stats[key]['real']
            stats[key]['expected']= 0 
            exp = -1
            while x > 0:
                if stats[key]['expected'] % 8 == 0:
                    exp += 1
                stats[key]['expected'] += 1 /(2 ** exp)
                x -= 1 

        stats[key]['mean'] = statistics.mean(stats[key]['occurrence'])
        stats[key]['stdev'] = statistics.stdev(stats[key]['occurrence'])
        stats[key]['variance'] = statistics.variance(stats[key]['occurrence'])
        stats[key]['mean_abs_dev'] = statistics.mean(abs(x - stats[key]['mean']) for x in stats[key]['occurrence']) 
        stats[key]['max_abs_error'] = max(abs(x - stats[key]['expected']) for x in stats[key]['occurrence'])
        stats[key]['mean_abs_error'] = statistics.mean(abs(x - stats[key]['expected']) for x in stats[key]['occurrence']) 
        stats[key]['mean_rel_error'] = statistics.mean(abs(x - stats[key]['expected'])/stats[key]['expected'] for x in stats[key]['occurrence'])
        stats[key]['mean_acc_ratio'] = statistics.mean(x/stats[key]['expected'] for x in stats[key]['occurrence'])
        stats[key]['smallest'] = min(stats[key]['occurrence'])
        stats[key]['largest'] = max(stats[key]['occurrence'])

    stats['avg_counter_size'] = sys.getsizeof(counter_dict['counter'])/len(counter_dict['counter'].keys())
    stats['real_size'] = sys.getsizeof(real_counter)

    # Saving results to a json file
    with open(f'{book_name}_{counter_dict["type"]}_stats.json', "w") as f:
        json.dump(stats, f, indent=4)

    counter_dict['stats'] = stats

    return counter_dict

def createBarPlot(counter, top_word_number, file_name):
    """Creates bar plot from counter

    Args:
        counter (counter): counter with words and number of occurrences
        top_word_number (int): Number of words to be displayed in graph
        file_name (str): Name of file to save the plot
    """
    # Find top words
    top_words = [(word, occurrence) for idx, (word, occurrence) in enumerate(counter.most_common()) if idx < top_word_number]
    word_list = [tup[0] for tup in top_words]
    occurrence_list = [tup[1] for tup in top_words]
    
    # Create and save figure
    px = 1/plt.rcParams['figure.dpi']
    figure = plt.figure(1, figsize=(720*px, 950*px))
    plt.barh(word_list, occurrence_list, color='firebrick')
    plt.gca().invert_yaxis()
    figure.savefig(f'../report/figs/{file_name}.pdf')
    figure.clear(True)


def createAverageCounter(counter_dict):
    """Create an average counter from a set of counters

    Args:
        counter_dict (dict): Dictionary of counters, with keys from 0 to k

    Returns:
        counter: Counter with average values
    """
    tmp_counter = Counter()
    for i in counter_dict['counter'].keys():
        tmp_counter += counter_dict['counter'][i]
    final_counter = Counter({key: value / len(counter_dict['counter'].keys()) for key, value in tmp_counter.items()})
    return final_counter
    



def main():
    # Define argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--books", help='List of names of books (.epub) to analyse.', type=list, default=['2004.epub', '2005.epub', '2010.epub'])
    ap.add_argument("-e", "--elements", help='String of elements, such as punctuation, to remove from the text.', type=str, default='.,?!"\'|:;()-—™“')
    ap.add_argument("-s", "--stop_words", help='String of the path of the location of the stop words txt file.', type=str, default='stop_words_english.txt')
    ap.add_argument("-p", "--probability", help='Fixed probabilistic counter probability (< 1)', type=int, default=0.25)
    ap.add_argument("-c", "--csuros", help='Dictionary of parameters to be used by Csuros counter - {"d": int, "p": int}', type=dict, default={'d' : 3, 'p' : 1/2})
    ap.add_argument("-r", "--repetitions", help='Number of repetitions of the counters', type=int, default=10)
    ap.add_argument("-t", "--top_words", help='Number of top words to be presented in the bar graphs', type=int, default=20)
    
    # Defining args
    args = vars(ap.parse_args())
    logger.info(f'The inputs of the script are: {args}')

    # Defining seed
    seed(88939)

    # Getting text from epub
    book_dict = dict([(key, dict()) for key in args['books']])
    for book in args['books']:
        book_dict[book]['text'] = readEPub(book)
    logger.info('Retrieved text for every desired epub')

    # Counting
    for book in book_dict.keys():
        book_dict[book]['total_count'] = totalCounter(book_dict[book]['text'], args['elements'], args['stop_words'])
        logger.info(f'Full counter of {book} carried out successfully')
        
        book_dict[book]['fixed_count'] = {'parameters': {'p': args['probability']}, 'type': 'fixed', 'counter' : dict()}
        book_dict[book]['csuros_count'] = {'parameters': args['csuros'], 'type': 'csuros', 'counter' : dict()}
        for i in range(args['repetitions']):
            book_dict[book]['fixed_count']['counter'][str(i)] = fixedProbabilisticCounter(book_dict[book]['text'], args['elements'], args['stop_words'], args['probability'])
            # logger.info(f'Fixed probability counter number {i} of {book} carried out successfully')

            book_dict[book]['csuros_count']['counter'][str(i)] = csurosCounter(book_dict[book]['text'], args['elements'], args['stop_words'], args['csuros'])
            # logger.info(f'Csuros counter number {i} of {book} carried out successfully')
        logger.info(f'Approximate counters of {book} done!')


    # Generate stats
    for book in book_dict.keys():
        book_dict[book]['fixed_count'] = createStats(book_dict[book]['fixed_count'], book_dict[book]['total_count'], args['top_words'], book)
        book_dict[book]['csuros_count'] = createStats(book_dict[book]['csuros_count'], book_dict[book]['total_count'], args['top_words'], book)

    # Create average graphs
    for book in book_dict.keys():
        book_dict[book]['fixed_count']['average'] = createAverageCounter(book_dict[book]['fixed_count'])
        book_dict[book]['csuros_count']['average'] = createAverageCounter(book_dict[book]['csuros_count'])

    print(book_dict[book]['fixed_count']['average'].most_common(10))
    # Generate bar graphs
    for book in book_dict.keys():
        createBarPlot(book_dict[book]['total_count'], args['top_words'], f'{book}-total')
        createBarPlot(book_dict[book]['fixed_count']['average'], args['top_words'], f'{book}-fixed-{args["repetitions"]}')
        createBarPlot(book_dict[book]['csuros_count']['average'], args['top_words'], f'{book}-csuros-{args["repetitions"]}')


if __name__ == '__main__':
    main()
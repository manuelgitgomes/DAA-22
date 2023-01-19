#!/usr/bin/env python3

import argparse
import logging
import ebooklib
from ebooklib import epub 
from bs4 import BeautifulSoup
from collections import Counter

# create logger
logger = logging.getLogger("PA1")
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


def main():
    # Define argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--number", help='Number of times to run the probabilistic counters.', type=int, default=10)
    ap.add_argument("-b", "--books", help='List of names of books (.epub) to analyse.', type=list, default=['2004.epub', '2005.epub', '2010.epub'])
    ap.add_argument("-e", "--elements", help='String of elements, such as punctuation, to remove from the text.', type=str, default='.,?!"\'|:;()-—')
    ap.add_argument("-s", "--stop_words", help='String of the path of the location of the stop words txt file.', type=str, default='stop_words_english.txt')
    
    # Defining args
    args = vars(ap.parse_args())
    logger.info(f'The inputs of the script are: {args}')

    # Getting text from epub
    text_dict = dict([(key, None) for key in args['books']])
    for book in args['books']:
        text_dict[book] = readEPub(book)
    logger.info('Retrieved text for every desired epub')

    for book in text_dict.keys():
        total_count = totalCounter(text_dict[book], args['elements'], args['stop_words'])
        logger.info(f'Full counter of {book} carried out successfully')
    


if __name__ == '__main__':
    main()
from Postprocessing import phrasefinder

'''
See: http://phrasefinder.io/documentation
query: query string

?	Inserts exactly one word. 
*	Inserts zero or more words.
/	To be read as an "or", which checks for the presence of the left- and right-hand side word. This operator can be chained.	
+	Completes a word prefix. At least the first three letters must be given.
"…"	Groups several words together that should be treated as a phrase. This operator is only useful in conjunction with the / operator.	

# topk: amount of results to return
# nmin: min length of matching phrases
# nmax: max length of matching phrases
'''


def query_google(query, topk=100, nmin=1, nmax=5):
    """
    Queries the google n-grams with API calls to Phrasefinder.
    :param query: The query
    :param topk: Max amount of results
    :param nmin: Minimum length of matching phrases
    :param nmax: Maximum length of matching phrases
    :return: A list of Phrase instances. Each phrase has a probability score.
    """
    options = phrasefinder.Options()
    options.nmin = nmin
    options.nmax = nmax
    options.topk = topk

    # Perform a request.
    result = phrasefinder.search(query, options)
    if result.status != phrasefinder.Status.Ok:
        print('Request was not successful: {}'.format(result.status))
        return
    return result.phrases


def n_gram_model(sentence, words, n=3):
    """
    Calculate the probabilities of every word given the n previous words in the sentence.
    Example usage:
    n_gram_model('The bridesman loves the', ['girl', 'man', 'sheep'], n=2)
    Returns: [('man', 0.56), ('girl', 0.39), ('sheep', 0.05)]
    Punctuation needs spaces.
    :param sentence: A sequence of previous words.
    :param words: A list of possible words which follow the sentence.
    :param n: The amount of previous words to consider.
    :return: A dictionary with words as key and score as value.
    """
    previous_words = ' '.join(sentence[-n:])
    query = previous_words + ' ' + ' / '.join(words)
    '''
    Give a default value of 0 as probability.
    If Google dataset doesn't contain the given n-gram then the probability will remain 0. 
    Note however, even if the dataset does not contain the n-gram, the construct is not impossible to occur. 
    '''
    results = dict((word, 0.0) for word in words)
    for phrase in query_google(query):
        prefix = ' '.join(token.text for token in phrase.tokens[:-1])
        word = phrase.tokens[-1].text
        # Variations in uppercase can occur. We have to check if the sentences are actually equal.
        if prefix == previous_words and word in words:
            results[word] = phrase.score
    return results

#print(n_gram_model(['She', 'loves', 'the'], ['girl', 'boy', 'sheep'], n=3))
#print(n_gram_model('He loves the', ['girl', 'boy', 'sheep'], n=3))
#print(n_gram_model('She loves the', ['woman', 'man', 'sheep'], n=2))


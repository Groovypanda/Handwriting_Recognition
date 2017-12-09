'''
Postprocessing consists of 2 steps:
1) Find the most likely word given a list of probabilities for every character in the word.
   This list contains the probability for a character (image) to be equal to that given character.
   We use an English dictionary to find possible likely words.
2) Find the most likely word in a certain context.
   Given a list of likely words (see previous step), we can find the most likely word based on the context.
   We use Markov models, n-gram models in particular to solve this problem.

These 2 techniques are often used for voice recognition and other techniques, but can be applied for this project aswell
to handle the high error rate of character recognition (+/- 20%).
'''


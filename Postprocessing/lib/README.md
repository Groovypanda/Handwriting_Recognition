# PhraseFinder Python Client

The official Python client for the [PhraseFinder](http://phrasefinder.io) web service

* [Documentation](https://mtrenkmann.github.io/phrasefinder-client-python/)

## Demo

```python
from __future__ import print_function
import phrasefinder

def main():

    # Set up your query.
    query = 'I like ???'

    # Set the optional parameter topk to 10.
    options = phrasefinder.Options()
    options.topk = 10

    # Perform a request.
    try:
        result = phrasefinder.search(query, options)
        if result.status != phrasefinder.Status.Ok:
            print('Request was not successful: {}'.format(result.status))
            return

        # Print phrases line by line.
        for phrase in result.phrases:
            print("{0:6f}".format(phrase.score), end="")
            for token in phrase.tokens:
                print(' {}_{}'.format(token.text, token.tag), end="")
            print()
        print('Remaining quota: {}'.format(result.quota))

    except Exception as error:
        print('Some error occurred: {}'.format(error))


if __name__ == '__main__':
    main()
```

## Clone and run

```sh
git clone https://github.com/mtrenkmann/phrasefinder-client-python.git
cd phrasefinder-client-python
python src/demo.py
```

## Output

```
0.175468 I_0 like_0 to_1 think_1 of_1
0.165350 I_0 like_0 to_1 think_1 that_1
0.149246 I_0 like_0 it_1 ._1 "_1
0.104326 I_0 like_0 it_1 ,_1 "_1
0.091746 I_0 like_0 the_1 way_1 you_1
0.082627 I_0 like_0 the_1 idea_1 of_1
0.064459 I_0 like_0 that_1 ._1 "_1
0.057900 I_0 like_0 it_1 very_1 much_1
0.055201 I_0 like_0 you_1 ._1 "_1
0.053677 I_0 like_0 the_1 sound_1 of_1
Remaining quota: 99
```

## Installation

Copy the file `src/phrasefinder.py` into the source directory of your project.

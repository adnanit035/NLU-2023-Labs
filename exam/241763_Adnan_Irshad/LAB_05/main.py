from pcfg import PCFG
import nltk


if __name__ == "__main__":
    # Sentences of my choice.
    print("*****1. My sentences:*****")
    sentences = [
        'The cat chased mouse',
        'The sun set over horizon',
    ]
    print(sentences)

    print("\n\n*****2. PCFG grammar rules for the sentences:*****")
    # PCFG grammar rules for the sentences.
    rules = [
        'S -> NP VP [1.0]',
        'NP -> Det N [0.5] | N [0.5]',
        'VP -> V NP [0.5] | V PP [0.5]',
        'PP -> P NP [1.0]',
        "Det -> 'The' [1.0]",
        'N -> "cat" [0.25] | "mouse" [0.25] | "sun" [0.25] | "horizon" [0.25]',
        'V -> "chased" [0.5] | "set" [0.5]',
        'P -> "over" [1.0]'
    ]
    print(rules)

    # PCFG model
    grammar = PCFG.fromstring(rules)

    # Validation of grammar by parsing the sentences with ViterbiParser
    print("\n\n*****3. Validation of grammar by parsing the sentences with ViterbiParser:*****")
    parser = nltk.ViterbiParser(grammar)
    for sentence in sentences:
        print(sentence)
        for tree in parser.parse(sentence.split()):
            tree.pretty_print()
            print()

    print("\n\n*****4. Generate 10 sentences using a PCFG by experimenting with nltk.parse.generate.generate:*****")
    # Then, generate 10 sentences using a PCFG by experimenting with nltk.parse.generate.generate
    for sent in grammar.generate(10):
        print(sent)

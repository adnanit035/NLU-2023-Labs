import spacy
import spacy_stanza
import stanza
from nltk.corpus import dependency_treebank
from nltk.parse import DependencyEvaluator
from nltk.parse.dependencygraph import DependencyGraph
from spacy.tokenizer import Tokenizer

# Download the stanza model if necessary
stanza.download("en")


if __name__ == "__main__":
    # getting the last 100 sentences from the dependency treebank
    sentences_ = dependency_treebank.sents()[-100:]

    sentences = []
    for sentence in sentences_:
        sentences.append(" ".join(sentence))

    #############################################################################
    # Downloading and Setting up the Spacy pipeline
    #############################################################################
    # Load the Spacy model
    nlp_spacy = spacy.load("en_core_web_sm")

    # Set up the conll formatter
    spacy_config = {
        "ext_names": {
            "conll_pd": "pandas"
        },
        "conversion_maps": {
            "deprel": {
                "nsubj": "subj"
            }
        }
    }

    # Add the formatter to the pipeline
    nlp_spacy.add_pipe("conll_formatter", config=spacy_config, last=True)

    # Split by white space
    nlp_spacy.tokenizer = Tokenizer(nlp_spacy.vocab)

    ############################################################################
    # Downloading and Setting up the Stanza pipeline
    ############################################################################
    # Set up the conll formatter
    # tokenize_pre-tokenized used to tokenize by whitespace
    nlp_stanza = spacy_stanza.load_pipeline("en", verbose=False, tokenize_pretokenized=True)

    stanza_config = {
        "ext_names": {
            "conll_pd": "pandas"
        },
        "conversion_maps": {
            "deprel": {
                "nsubj": "subj",
                "root": "ROOT"
            }
        }
    }

    # Add the formatter to the stanza pipeline
    nlp_stanza.add_pipe("conll_formatter", config=stanza_config, last=True)

    # Parse the sentences and create DependencyGraph objects
    graphs_spacy = []
    graphs_stanza = []
    for sentence in sentences:
        # Parse the sentence using Spacy
        doc_spacy = nlp_spacy(sentence)
        df_spacy = doc_spacy._.pandas
        tmp_spacy = df_spacy[["FORM", "XPOS", "HEAD", "DEPREL"]].to_string(header=False, index=False)
        # tmp_spacy = df_spacy[['FORM', 'UPOS', 'HEAD', 'DEPREL']].to_string(header=False, index=False)
        # print(f"Spacy parse of '{sentence}':\n{tmp_spacy}")
        try:
            graph_spacy = DependencyGraph(tmp_spacy)
            graphs_spacy.append(graph_spacy)
        except Exception as e:
            print(f"Failed to create DependencyGraph for Spacy parse of '{sentence}': {e}")

        # # Parse the sentence using Stanza
        doc_stanza = nlp_stanza(sentence)
        df_stanza = doc_stanza._.pandas
        tmp_stanza = df_stanza[["FORM", "XPOS", "HEAD", "DEPREL"]].to_string(header=False, index=False)
        # tmp_stanza = df_stanza[['FORM', 'UPOS', 'HEAD', 'DEPREL']].to_string(header=False, index=False)
        # print(f"Stanza parse of '{sentence}':\n{tmp_stanza}")
        try:
            graph_stanza = DependencyGraph(tmp_stanza, top_relation_label='root')
            graphs_stanza.append(graph_stanza)
        except Exception as e:
            print(f"Failed to create DependencyGraph for Stanza parse of '{sentence}': {e}")

    # Evaluate the Spacy and Stanza parses
    # Create DependencyEvaluator objects
    evaluator_spacy = DependencyEvaluator(graphs_spacy, dependency_treebank.parsed_sents()[-100:])
    evaluator_stanza = DependencyEvaluator(graphs_stanza, dependency_treebank.parsed_sents()[-100:])

    # Print LAS and UAS for each parser
    las_spacy, uas_spacy = evaluator_spacy.eval()
    las_stanza, uas_stanza = evaluator_stanza.eval()
    print(f"Spacy: LAS={las_spacy}, UAS={uas_spacy}")
    print(f"Stanza: LAS={las_stanza}, UAS={uas_stanza}")

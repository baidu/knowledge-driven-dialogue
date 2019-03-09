                                       Knowledge-driven Dialogue
----------------------------------------------------------------------------------------------------------------
This is a paddlepaddle implementation of retrieval-based model for knowledge-driven dialogue


Requirements
-----------------------------------------------
cuda=9.0
cudnn=7.0
python>=2.7
numpy
paddlepaddle>=1.3


Quickstart
-----------------------------------------------
Step 1: Train: support 3 mode, match、match_kn and match_gene
	mode1: match, input is history and response
		   sh run_train.sh match
	mode2: match_kn, input is history, response, chat_path, knowledge
		   sh run_train.sh match_kn
	mode3: match_kn, input is history, response, chat_path, knowledge
	       and generalizes target_a/target_b of goal for all inputs, replaces them with slot mark
		   sh run_train.sh match_kn_gene

Step 2: Test
test model with the following commands.
	sh run_predict.sh match|match_kn|match_kn_gene


Note:
    1. the script run_train.sh/run_test.sh shows all the processes including data processing and model training/testing.
       be sure to read it carefully and follow it.

    2. the most important step in run_train.sh/run_test.sh is to build candidates by ./tools/construct_candidate.py.
       you have to reimplement the core function yourself: get_candidate_for_conversation





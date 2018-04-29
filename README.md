# IBM Model 1 and IBM Model 2

Use:

$ python mt.py

to see the command line usage of this program.

Specifically, if you want to train IBM Model 1, use:

$ python my.py [target_corpus_file] [source_corpus_file] model1

This command generates 5 .ibm1 files: 'params_i.ibm1' where i is the ith iteration. 
For example, params_5.ibm1 stores the t parameters after 5 iterations of EM 
Algorithm in IBM Model 1.

If you want to run test on IBM Model 1, use:

$ python mt.py [target_corpus_test_file] [source_corpus_test_file] [parameter_file] [output_file] model1

This will store the alignments produced to [output_file] which could be evaluated
using the eval_alignment.py command:

$ python eval_alignment.py [key_file] [output_file]

If you want to train IBM Model 2, use:

$ python mt.py [target_corpus_file] [source_corpus_file] model2

This command trains the IBM Model 2 using the EM Algorithm and runs for 5 iterations.
The initial t parameters are extracted from the file "params_5.ibm1" so make sure
you train IBM Model 1 first before training IBM Model 2.

If you want to test IBM Model 2, use:

$ python mt.py [target_test_file] [source_test_file] [i] [output_file] model2

where i indicates which set of q and t parameters you wish to use. For example,
when i = 4 it means you will use the parameters after 4 iterations of EM
Algorithm. Make sure you trained IBM Model 2 already before running this command.

You could run

$ python grow.py 

and get an intersection of alignments output in file "grown.out"

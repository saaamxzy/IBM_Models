import sys
import time
from IBM1 import IBM1
from IBM2 import IBM2




if __name__ == '__main__':

	if len(sys.argv) == 4:
		if sys.argv[3] == '1':
			ibm1 = IBM1()
			ibm1.init_train()
			ibm1.parallel_read(sys.argv[1], sys.argv[2])
			ibm1.initialize()
			ibm1.save_n('n.ibm1')
			for i in xrange(5):
				ibm1.em()
				ibm1.save_params('params_{}.ibm1'.format(i+1))
		elif sys.argv[3] == '2':
			ibm2 = IBM2()
			ibm2.init_train('params_5.ibm1')
			ibm2.parallel_read(sys.argv[1], sys.argv[2])
			ibm2.read_params()
			for i in xrange(5):
				ibm2.em()
				ibm2.save_params('params_{}.ibm2'.format(i+1), 'q_params_{}.ibm2'.format(i+1))

			
	elif len(sys.argv) == 6:
		if sys.argv[5] == '1':
			ibm1 = IBM1()
			ibm1.init_test(sys.argv[3], 'corpus.en', 'corpus.es')
			ibm1.read_params()
			# ibm1.save_params('params_6.ibm1')
			ibm1.predict(sys.argv[2], sys.argv[1], sys.argv[4])
		elif sys.argv[5] == '2':
			ibm2 = IBM2()
			ibm2.init_test(sys.argv[3], 'corpus.en', 'corpus.es')
			ibm2.read_params()
			ibm2.read_q('q_params_5.ibm2')
			ibm2.predict(sys.argv[2], sys.argv[1], sys.argv[4])

	else:
		usage = \
		'Argument number incorrect. Use\n' +\
		'$ python mt.py [english_corpus_training] [spanish_corpus_training] [model_number(1 or 2)] for training\n' +\
		'or\n' +\
		'$ python mt.py [english_corpus_test] [spanish_corpus_test] [parameter_file] [output_file] [model_number(1 or 2)] for testing\n'
		print(usage)
		exit()

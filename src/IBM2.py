from collections import defaultdict
import time



class IBM2():
	def __init__(self):
		'''
		probability p(f|e)
		'''
		self.t = defaultdict(lambda: defaultdict(float))
		self.q = defaultdict(lambda: defaultdict(float))
		self.is_q_init = defaultdict(bool)

		

	def init_train(self, param_file):
		self.param_file = param_file
		# self.en_vocab = set()
		# self.es_vocab = set()
		self.counts = defaultdict(float)
		self.counts_q1 = defaultdict(lambda: defaultdict(float))
		self.counts_q2 = defaultdict(float)
		#self.n = defaultdict(float)

		self.en_corpus = None
		self.es_corpus = None

	def init_test(self, param_file, en, es):
		self.param_file = param_file

		fen = open(en, 'r')
		self.en_corpus = fen.readlines()
		fen.close()

		fes = open(es, 'r')
		self.es_corpus = fes.readlines()
		fes.close()

	def read_params(self):
		in_file = open(self.param_file, 'r')

		for i,line in enumerate(in_file):
			if i % 100000 == 0:
				print (i)
			params = line.strip().split('($$)')
			f = params[0]
			e = params[1]
			value = eval(params[2])
			# if f not in self.t:
			# 	self.t[f] = dict()
			self.t[f][e] = value

		in_file.close()





	def parallel_read(self, en, es):

		print('reading corpus')

		start = time.time()

		fen = open(en, 'r')
		en_corpus = fen.readlines()
		fen.close()

		fes = open(es, 'r')
		es_corpus = fes.readlines()
		fes.close()

		assert len(en_corpus) == len(es_corpus), "parallel corpus length mismatch"

		self.en_corpus = en_corpus
		self.es_corpus = es_corpus


		time_elapsed = time.time() - start
		print('time spent in read: ' + str(time_elapsed))


	def get_n(self, word):
		return self.n[word]

	def get_t(self, f, e):
		result = self.t[f][e]
		if result == 0.0:
			print('method get_t: no f|e found in t table. f = {}, e = {}.'.format(f,e))

		return result


	def get_q(self, j, i, l, m):
		# if j not in self.q:
		# 	self.q[j] = dict()
		result = self.q[j][(i,l,m)]

		if result == 0.0 and not self.is_q_init[(j,i,l,m)]:
			self.q[j][(i,l,m)] = 1.0 / (l + 1.0)
			self.is_q_init[(j,i,l,m)] = True
		return self.q[j][(i,l,m)]
		# if j not in self.q:
		# 	self.q[j] = dict()

		# if (i,l,m) in self.q[j]:
			
		# 	return self.q[j][(i,l,m)]
		# else:
		# 	self.q[j][(i,l,m)] = 1.0 / (l + 1.0)
		# 	return self.q[j][(i,l,m)]

	def set_q(self, j, i, l, m, value):
		# if j not in self.q:
		# 	self.q[j] = dict()
		self.q[j][(i,l,m)] = value



	def delta_2(self, j, i, lk, mk, es_sentence, en_sentence):
		fi = es_sentence[i]
		ej = en_sentence[j]
		numerator = self.get_q(j, i, lk, mk) * self.get_t(fi, ej)

		denominator = 0.0
		
		for temp_j in xrange(lk+1):
			temp_ej = en_sentence[temp_j]
			denominator += self.get_q(temp_j, i, lk, mk) * self.get_t(fi, temp_ej)

		result = numerator / denominator

		return result

	def em(self):
		'''
		expectation-maximization algorithm
		'''
		self.counts = defaultdict(float)
		self.counts_q1 = defaultdict(lambda: defaultdict(float))
		self.counts_q2 = defaultdict(float)
		print('length of corpus: ' + str(len(self.en_corpus)))
		start = time.time()
		for k in xrange(len(self.en_corpus)):
			if k % 500 == 0:
				print(str(len(self.en_corpus) - k) + ' sentences left')
			es_sentence = self.es_corpus[k].split()
			en_sentence = ['NULL'] + self.en_corpus[k].split()
			mk = len(es_sentence)
			lk = len(en_sentence) - 1
			for i in xrange(mk):
				for j in xrange(lk+1):
					ej = en_sentence[j]
					fi = es_sentence[i]
					delta = self.delta_2(j, i, lk, mk, es_sentence, en_sentence)
					self.counts[(ej, fi)] += delta
					self.counts[ej] += delta
					self.counts_q1[j][(i,lk,mk)] += delta
					self.counts_q2[(i,lk,mk)] += delta

		for f in self.t.keys():
			for e in self.t[f].keys():
				self.t[f][e] = self.counts[(e, f)] / self.counts[e]

		for j in self.q.keys():
			for ilm in self.q[j].keys():
				i = ilm[0]
				l = ilm[1]
				m = ilm[2]
				self.set_q(j, i,l,m, self.counts_q1[j][ilm] / self.counts_q2[ilm])

		time_elapsed = time.time() - start
		print('time spent in em: ' + str(time_elapsed))


	def read_q(self, qfname):
		qf = open(qfname, 'r')
		
		for i,line in enumerate(qf):
			if i % 100000 == 0:
				print (i)
			params = line.strip().split('($$)')
			j = eval(params[0])
			ilm = eval(params[1])
			value = eval(params[2])
			
			self.q[j][ilm] = value
			
		qf.close()

	def get_alignment(self, es_sentence, en_sentence, out_file, index):
		#print(index)
		l = len(en_sentence)-1
		m = len(es_sentence)
		for i in xrange(m):
			max_index = -1
			max_val = 0
			fi = es_sentence[i]
			for j in xrange(l+1):
				ej = en_sentence[j]

				temp_val = self.get_t(fi,ej) * self.q[j][(i,l,m)]
				if temp_val > max_val:
					max_val = temp_val
					max_index = j
			if max_index != -1:
				out_file.write('{} {} {}\n'.format(index+1, max_index, i+1))

	def predict(self, es_fname, en_fname, out_fname):
		es_file = open(es_fname, 'r')
		en_file = open(en_fname, 'r')
		out_file = open(out_fname, 'w')
		es_sentences = es_file.readlines()
		en_sentences = en_file.readlines()

		assert len(es_sentences) == len(en_sentences), 'In predict: corpus length mismatch'
		print('starting to predict')
		for i in xrange(len(es_sentences)):
			self.get_alignment(es_sentences[i].split(), ['NULL'] + en_sentences[i].split(), out_file, i)

		es_file.close()
		en_file.close()
		out_file.close()

	def save_params(self, t_out_fname, q_out_fname):
		out_f = open(t_out_fname, 'w')
		for f in self.t.keys():
			for e in self.t[f].keys():
				out_f.write('{}($$){}($$){}\n'.format(f,e,self.t[f][e]))

		out_f.close()

		out_f = open(q_out_fname, 'w')
		for j in self.q.keys():
			for ilm in self.q[j]:
				out_f.write('{}($$){}($$){}\n'.format(j,ilm,self.q[j][ilm]))
		out_f.close()

	def save_n(self, out_fname):
		out_f = open(out_fname, 'w')
		for each in self.n:
			out_f.write('{}(**){}\n'.format(each, self.n[each]))
		out_f.close()
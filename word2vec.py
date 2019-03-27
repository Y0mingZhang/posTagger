from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

def keyed_vector():
	tmp_file = get_tmpfile("/tmp/word2vec.txt")
	try:
		print('Loading Glove File...')
		model = KeyedVectors.load_word2vec_format(tmp_file)
	except:
		# Fix glove file path
		glove_file = datapath('/Users/jedimaster/Downloads/glove/glove.6B.200d.txt')
		_ = glove2word2vec(glove_file, tmp_file)
		model = KeyedVectors.load_word2vec_format(tmp_file)
	return model
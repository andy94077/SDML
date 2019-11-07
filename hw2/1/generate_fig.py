import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def draw(title=None, xlabel=None, ylabel=None, has_legend=False,savefig=False):
	if title is not None:
		plt.title(title)
	if xlabel is not None:
		plt.xlabel(xlabel)
	if ylabel is not None:
		plt.ylabel(ylabel)
	if has_legend:
		plt.legend()
	if savefig is False:
		plt.show()
	else:
		plt.savefig(savefig)
	plt.close()

df = pd.read_csv(sys.argv[1])
t = range(1, df['epoch'].shape[0] + 1)

# loss
plt.plot(t,df['loss'], label='train_total_loss')
plt.plot(t,df['val_loss'], label='valid_total_loss')
plt.plot(t,df['decoder_out_loss'], label='train_sentence_loss')
plt.plot(t,df['val_decoder_out_loss'], label='valid_sentence_loss')
plt.plot(t,df['word_out_loss'], label='train_word_loss')
plt.plot(t,df['val_word_out_loss'], label='valid_word_loss')
plt.xticks(t)
draw(title='loss', xlabel='epoch', has_legend=True, savefig=os.path.basename(sys.argv[1][:sys.argv[1].rfind('.')])+'_loss.png')

# acc
plt.plot(t,df['decoder_out_sparse_categorical_accuracy'], label='train_sentence_acc')
plt.plot(t,df['val_decoder_out_sparse_categorical_accuracy'], label='valid_sentence_acc')
plt.plot(t,df['word_out_sparse_categorical_accuracy'], label='train_word_acc')
plt.plot(t,df['val_word_out_sparse_categorical_accuracy'], label='valid_word_acc')
plt.xticks(t)
draw(title='accuracy', xlabel='epoch', has_legend=True, savefig=os.path.basename(sys.argv[1][:sys.argv[1].rfind('.')])+'_acc.png')


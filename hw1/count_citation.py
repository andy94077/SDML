import os, sys
import numpy as np
import pandas as pd

data_dir = sys.argv[1]
train_or_test = sys.argv[2]

if train_or_test == 'train':
	Y = np.load(os.path.join(data_dir, 'task2_trainY.npy'))
	id_paper = pd.read_csv(os.path.join(data_dir, 'mag_paper_data/id_paperId.tsv'), sep='\t', encoding='utf-8', nrows=Y.shape[0])
	citation_graph = pd.read_csv(os.path.join(data_dir, 'mag_paper_data/citation_graph.tsv'), sep='\t', encoding='utf-8')

	m = {row.PaperId: row.Id for row in id_paper.itertuples(index=False)}
	citation_count = np.zeros_like(Y)

	for row in citation_graph.itertuples(index=False):
		if row.PaperId in m and row.CitedPaperId in m:
			citation_count[int(m[row.PaperId][1:])-1] += Y[int(m[row.CitedPaperId][1:])-1]

	citation_count = np.nan_to_num(citation_count/np.sum(citation_count, axis=1, keepdims=True), copy=False)
	np.save(os.path.join(data_dir, 'citation_count_train.npy'), citation_count)
else:
	Y = np.load(os.path.join(data_dir, 'task2_trainY.npy'))
	testX = np.load(os.path.join(data_dir, 'task2_testX.npy'))
	id_paper = pd.read_csv(os.path.join(data_dir, 'mag_paper_data/id_paperId.tsv'), sep='\t', encoding='utf-8', nrows=Y.shape[0]+testX.shape[0])

	citation_graph = pd.read_csv(os.path.join(data_dir, 'mag_paper_data/citation_graph.tsv'), sep='\t', encoding='utf-8')

	train_m = {row.PaperId: row.Id for row in id_paper.iloc[:Y.shape[0]].itertuples(index=False)}
	test_m = {row.PaperId: row.Id for row in id_paper.iloc[Y.shape[0]:].itertuples(index=False)}
	citation_count = np.zeros((testX.shape[0], 4), dtype=np.float64)

	for row in citation_graph.itertuples(index=False):
		if row.PaperId in test_m and row.CitedPaperId in train_m:
			citation_count[int(test_m[row.PaperId][1:])-1] += Y[int(train_m[row.CitedPaperId][1:])-1]

	citation_count = np.nan_to_num(citation_count/np.sum(citation_count, axis=1, keepdims=True), copy=False)
	np.save(os.path.join(data_dir, 'citation_count_test.npy'), citation_count)

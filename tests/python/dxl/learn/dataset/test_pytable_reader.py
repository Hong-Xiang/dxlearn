from typing import NamedTuple, Optional, Dict
from pathlib import Path
import tables as tb
import attr
import typing
import tensorflow as tf
import pytest
import numpy as np
from dxl.learn.dataset.data_column import PytableData,PytableReader
import tempfile as t

class train_part(tb.IsDescription):
	train_x = tb.Float32Col(shape=(32,32,1))
	train_y = tb.Float32Col()
class test_part(tb.IsDescription):
	test_x = tb.Float32Col(shape=(32,32,1))
	test_y = tb.Float32Col()
class Test_PytableReader:
	def create_test_file(self):
		temp_dir = t.TemporaryDirectory()
		f = tb.open_file(temp_dir.name + 'test_pytable_reader.h5', mode='w')
		root = f.root
		g1 = f.create_group(root,'group1')
		t1 = f.create_table(g1,'train',train_part)
		t2 = f.create_table(g1,'test',test_part)
		it1 = t1.row
		it2 = t2.row
		for i in range(10):
			it1['train_x']=(i+1)*np.ones((32,32,1), dtype=np.float32)
			it1['train_y']=np.random.random()
			it2['test_x']=(10+i+1)*np.ones((32,32,1), dtype=np.float32)
			it2['test_y']=np.random.random()
			it1.append()
			it2.append()
		t1.flush()
		t2.flush()
		f.close()
		file = temp_dir.name + 'test_pytable_reader.h5'
		return file

	def test_open(self):
		file = self.create_test_file()
		with PytableReader(file) as pr:
			assert pr.file is not None
			assert pr.file_path ==file
	def test_get_h5_to_table_with_name(self):
		file = self.create_test_file()
		with PytableReader(file,'r') as pr:
			pr.get_h5_to_table('/group1', table_name='train')
			assert pr.table is not None
			assert pr.table['train'] is not None
	def test_get_h5_to_table_with_dir(self):
		file = self.create_test_file()
		with PytableReader(file) as pr:
			pr.get_h5_to_table('/group1/train')
			assert pr.table is not None
			assert pr.table['train'] is not None
	def test_make_iterator(self):
		file = self.create_test_file()
		with PytableReader(file) as pr:
			tb1 = pr.get_h5_to_table('/group1/train')
			it = pr.make_iterator(tb1)
			cmp = [[[1] for i in range(32)]] * 32
			cmp = np.array(cmp)
			assert (next(it())[0] == cmp).all()
	def test_to_dataset(self):
		file = self.create_test_file()
		with PytableReader(file) as pr:
			tb1 = pr.get_h5_to_table('/group1/train')
			dt = pr.to_dataset('train')
			it = dt.make_one_shot_iterator()
			cmp = [[[1] for i in range(32)]] * 32
			cmp = np.array(cmp)
			with tf.Session() as sess:
				a = sess.run(it.get_next())
			assert (a[0] == cmp).all()
	def test_map_to_tf_type(self):
		file = self.create_test_file()
		with PytableReader(file) as pr:
			a = np.array([[1,2],[1,2],[1,2]],dtype=np.int16)
			assert pr.type_mapper[a.dtype] == tf.int16
	def  test_get_type_and_shape(self):
		file = self.create_test_file()
		with PytableReader(file) as pr:
			tb1 =pr.get_h5_to_table('/group1/train')
			t, s =pr.get_type_and_shape(tb1)
			assert t == (tf.float32, tf.float32)

	def test_to_tensor(self):
		file = self.create_test_file()
		with PytableReader(file) as pr:
			tb1 = pr.get_h5_to_table('/group1/train')
			dt = pr.to_dataset('train')
			it = dt.make_one_shot_iterator()
			data1 = pr.to_tensor(dt, iterator=it)
			assert isinstance(data1[0], tf.Tensor)
			assert isinstance(data1[1], tf.Tensor)
			data2 = pr.to_tensor(dt)
			assert isinstance(data2[0], tf.Tensor)
			assert isinstance(data2[1], tf.Tensor)
	def test_get_data(self):
		file =self.create_test_file()
		with PytableReader(file) as pr:
			pr.get_h5_to_table('/group1/train')
			dt1 = pr.to_dataset('train')
			it = dt1.make_one_shot_iterator()
			data = pr.get_data(dt1, iterator=it)
			with tf.Session() as sess:
				data = sess.run(data)
				assert data is not None
				cmp = [[[1] for i in range(32)]] * 32
				cmp = np.array(cmp)
				assert (data[0] == cmp).all()
	def test_process_dataset(self):
		file =self.create_test_file()
		with PytableReader(file) as pr:
			pr.get_h5_to_table('/group1/train')
			dt1 = pr.to_dataset('train')
			dt1 =pr.process_dataset(dt1,repeat=1,batch_size=3)
			data =pr.get_data(dt1)
			with tf.Session() as sess:
				data = sess.run(data)
				assert np.shape(data[0]) == (3,32,32,1)

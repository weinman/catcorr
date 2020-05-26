# test.py - Run unit tests on the operations

import tensorflow as tf
import numpy as np
import unittest
import catcorr

class TestRkCoeffNumPy(unittest.TestCase):

    def test_zeros(self): 
        zeros = np.zeros((4, 4))
        result = catcorr.rk_coeff_np(zeros)
        self.assertEqual(result, 0)

    def test_ones(self): 
        ones = np.ones((4, 4))
        result = catcorr.rk_coeff_np(ones)
        self.assertEqual(result, 0)

    def test_diag_ones(self):
        diag_ones = np.eye(7)
        result = catcorr.rk_coeff_np(diag_ones)
        self.assertEqual(result, 1)

    def test_arange(self):
        precalculated_result = -0.07007127538
        arange = np.arange(9).reshape((3,3))
        result = catcorr.rk_coeff_np(arange)
        difference = abs(precalculated_result - result)
        self.assertLess(difference, 0.00000001)
    
    def test_large_table(self):
        large_table = np.ones((500, 500))
        result = catcorr.rk_coeff_np(large_table)
        self.assertEqual(result, 0)

    def test_arange_2(self):
        precalculated_result = -0.031676277618
        arange = np.arange(16, dtype=np.float).reshape((4,4))
        result = catcorr.rk_coeff_np(arange)
        difference = abs(precalculated_result - result)
        self.assertLess(difference, 0.00000001)

    # The following values are taken from 'A Comparison of MCC and CEN
    # Error Measures' by G. Jurman, S. Riccadonna, and C. Furlanello
    # (2012)

    def test_inv_diagonal(self):
        precalculated_result = -0.333
        confusion_matrix = np.full((4,4), 5)
        np.fill_diagonal(confusion_matrix, 0)
        result = catcorr.rk_coeff_np(confusion_matrix)
        difference = abs(precalculated_result - result)
        self.assertLess(difference, 0.001)

    def test_col(self):
        confusion_matrix = np.zeros((4,4))
        confusion_matrix[:,1] = [15,15,15,15]
        result = catcorr.rk_coeff_np(confusion_matrix)
        self.assertEqual(result, 0)


        
class TestRkCoeffTensorFlow(unittest.TestCase):

    def setUp(self):
        self.sess = tf.compat.v1.Session()
    
    def test_zero(self):
        zeros = tf.constant(np.zeros((4, 4)))
        result_tensor = catcorr.rk_coeff_tf(zeros)
        result = self.sess.run(result_tensor)
        self.assertEqual(result, 0)

    def test_ones(self):
        ones = tf.constant(np.ones((4, 4)))
        result_tensor = catcorr.rk_coeff_tf(ones)
        result = self.sess.run(result_tensor)
        self.assertEqual(result, 0)

    def test_arange(self):
        precalculated_result = -0.07007127538
        arange = tf.constant(np.arange(9, dtype=np.float).reshape((3,3)))
        result_tensor = catcorr.rk_coeff_tf(arange)
        result = self.sess.run(result_tensor)
        difference = abs(precalculated_result - result)
        self.assertLess(difference, 0.00000001)

    def test_large_table(self):
        large_table = tf.constant(np.ones((500, 500)))
        result_tensor = catcorr.rk_coeff_tf(large_table)
        result = self.sess.run(result_tensor)
        self.assertEqual(result, 0)

    def test_arange_2(self):
        precalculated_result = -0.031676277618
        arange = tf.constant(np.arange(16, dtype=np.float).reshape((4,4)))
        result_tensor = catcorr.rk_coeff_tf(arange)
        result = self.sess.run(result_tensor)
        difference = abs(precalculated_result - result)
        self.assertLess(difference, 0.00000001)

    # The following values are taken from 'A Comparison of MCC and
    # CEN Error Measures' by G. Jurman, S. Riccadonna, and
    # C. Furlanello (2012)

    def test_inv_diagonal(self):
        precalculated_result = -0.333
        confusion_matrix = np.full((4,4), 5)
        np.fill_diagonal(confusion_matrix, 0)
        result_tensor = catcorr.rk_coeff_tf(tf.constant(confusion_matrix))
        result = self.sess.run(result_tensor)
        difference = abs(precalculated_result - result)
        self.assertLess(difference, 0.001)

    def test_col(self):
        confusion_matrix = np.zeros((4,4))
        confusion_matrix[:,1] = [15,15,15,15]
        result_tensor = catcorr.rk_coeff_tf(tf.constant(confusion_matrix))
        result = self.sess.run(result_tensor)
        self.assertEqual(result, 0)
        
    def tearDown(self):
        self.sess.close()
        

class TestSoftConfusionMatrixTensorFlow(unittest.TestCase):

    def setUp(self):
        self.sess = tf.compat.v1.Session()

    def test_identity_labels_vector(self):
        labels = tf.constant( np.arange(10) )
        probs = tf.constant( np.eye(10) )
        result_tensor = catcorr.soft_confusion_matrix_tf(labels, probs)
        result = self.sess.run(result_tensor)
        expected = np.eye(10)
        
        self.assertTrue( np.array_equal( result, expected ) )

    def test_identity_labels_matrix(self):
        labels = tf.constant( np.eye(10,dtype=np.int32) )
        probs = tf.constant( np.eye(10) )
        result_tensor = catcorr.soft_confusion_matrix_tf(labels, probs)
        result = self.sess.run(result_tensor)
        expected = np.eye(10)

        self.assertTrue( np.array_equal( result, expected ) )

    def tearDown(self):
        self.sess.close()

if __name__ == '__main__':
    unittest.main()

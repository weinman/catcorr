# CatCorr
# Copyright (c) 2020 Jerod Weinman and Nathan Gifford
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# test.py - Run unit tests on select operations

import tensorflow as tf
import numpy as np
import unittest
import catcorr.core as catcorr

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
        self.assertLess(difference, 1e-08)
    
    def test_large_table(self):
        large_table = np.ones((500, 500))
        result = catcorr.rk_coeff_np(large_table)
        self.assertEqual(result, 0)

    def test_arange_2(self):
        precalculated_result = -0.031676277618
        arange = np.arange(16, dtype=np.float).reshape((4,4))
        result = catcorr.rk_coeff_np(arange)
        difference = abs(precalculated_result - result)
        self.assertLess(difference, 1e-08)

    # The following values are taken from
    #   G. Jurman, S. Riccadonna, and C. Furlanello (2012). A Comparison
    #   of MCC and CEN Error Measures in Multi-Class Prediction. PLoS
    #   ONE 7(8): e41882. https://doi.org/10.1371/journal.pone.0041882

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
        result = catcorr.rk_coeff_tf(zeros)
        self.assertEqual(result, 0)

    def test_ones(self):
        ones = tf.constant(np.ones((4, 4)))
        result = catcorr.rk_coeff_tf(ones)
        self.assertEqual(result, 0)

    def test_arange(self):
        precalculated_result = -0.07007127538
        arange = tf.constant(np.arange(9, dtype=np.float).reshape((3,3)))
        result = catcorr.rk_coeff_tf(arange)
        difference = abs(precalculated_result - result)
        self.assertLess(difference, 1e-08)

    def test_large_table(self):
        large_table = tf.constant(np.ones((500, 500)))
        result = catcorr.rk_coeff_tf(large_table)
        self.assertEqual(result, 0)

    def test_arange_2(self):
        precalculated_result = -0.031676277618
        arange = tf.constant(np.arange(16, dtype=np.float).reshape((4,4)))
        result = catcorr.rk_coeff_tf(arange)
        difference = abs(precalculated_result - result)
        self.assertLess(difference, 1e-08)

    # The following values are taken from 'A Comparison of MCC and
    # CEN Error Measures' by G. Jurman, S. Riccadonna, and
    # C. Furlanello (2012)

    def test_inv_diagonal(self):
        precalculated_result = -0.333
        confusion_matrix = np.full((4,4), 5)
        np.fill_diagonal(confusion_matrix, 0)
        result = catcorr.rk_coeff_tf(tf.constant(confusion_matrix))
        difference = abs(precalculated_result - result)
        self.assertLess(difference, 0.001)

    def test_col(self):
        confusion_matrix = np.zeros((4,4))
        confusion_matrix[:,1] = [15,15,15,15]
        result = catcorr.rk_coeff_tf(tf.constant(confusion_matrix))
        self.assertEqual(result, 0)
        
    def tearDown(self):
        self.sess.close()
        

class TestSoftConfusionMatrixTensorFlow(unittest.TestCase):

    def setUp(self):
        self.sess = tf.compat.v1.Session()

    def test_identity_labels_vector(self):
        labels = tf.constant( np.arange(10) )
        probs = tf.constant( np.eye(10) )
        result = catcorr.soft_confusion_matrix_tf(labels, probs)
        expected = np.eye(10)
        
        self.assertTrue( np.array_equal( result, expected ) )

    def test_identity_labels_matrix(self):
        labels = tf.constant( np.eye(10,dtype=np.int32) )
        probs = tf.constant( np.eye(10) )
        result = catcorr.soft_confusion_matrix_tf(labels, probs)
        expected = np.eye(10)

        self.assertTrue( np.array_equal( result, expected ) )

    def tearDown(self):
        self.sess.close()

if __name__ == '__main__':
    unittest.main()

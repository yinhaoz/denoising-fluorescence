% =========================================================================
%                              KSVD - Toolbox
% =========================================================================

The K-SVD is a new algorithm for training dictionaries for linear
representation of signals. Given a set of signals, the K-SVD tries to
extract the best dictionary that can sparsely represent those signals. 

Thorough discussion concerning the K-SVD algorithm can be found in:
"The K-SVD: An Algorithm for Designing of Overcomplete Dictionaries for 
Sparse Representation", written by M. Aharon, M. Elad, and A.M. Bruckstein, 
and appeared in the IEEE Trans. On Signal Processing, Vol. 54, no. 11, 
pp. 4311-4322, November 2006. 

In this toolbox you can find the following files:
================================================

1. KSVD - the main file in this toolbox that implements the KSVD algorithm. Input and output parameters are described inside.
2. KSVD_NN - a variation of the KSVD algorithm for non-negative matrix factorization (non-negative dictionary and coefficients).
  
The following 3 files implements denoising according to 3 different methods described in "Image Denoising Via Sparse and Redundant 
           representations over Learned Dictionaries", appeared in the IEEE Trans. on Image Processing, Vol. 15, no. 12, 
           pp. 3736-3745, December 2006. 
===================================================================================================================================

3. denoiseImageDCT - denoising of an image using an overcomplete DCT dictionary.
4. denoiseImageGlobal - denoising of an image using a global trained dictionary. The global dictionary is stored in 
           the file 'globalTrainedDictionary.mat', which must exist in the directory. Alternatively, this function can be
           used for denoising of images using some other dictionary, for example, a dictionary that was trained by the 
           K-SVD algorithm, executed by the user.
5. denoiseImageKSVD - denoising of an image using a dictionary trained on noisy patches of the image.

The following 3 files are demo files that can be executed without any parameters,
================================================================================

6. demo1 - run file that executes synthetic test to validate the K-SVD algorithm (the same synthetic test that was presented in the paper).
7. demo2 - run file that executes denoising by 3 different methods, all described in "Image Denoising Via Sparse and Redundant 
           representations over Learned Dictionaries", appeared in the IEEE Trans. on Image Processing, Vol. 15, no. 12, 
           pp. 3736-3745, December 2006. 
8. demo3 - run file that executes synthetic test to validate the non-negative variation of the KSVD algorithm (the same test is presented
           in "K-SVD and its non-negative variant for dictionary design", written by M. Aharon, M. Elad, and A.M. Bruckstein 
           and appeared in the Proceedings of the SPIE conference wavelets, Vol.  5914, July 2005. 

The rest of the files assist the above files:
============================================

9. gererateSyntheticDictionaryAndData - Generates a random dictionary according to the parameters, and then generates signals as 
           sparse combinations of the atoms of this dictionary. Finally, it adds while Gaussian noise with a given s.d.
10. displayDictionaryElementsAsImage - displays the atoms of a dictionary as blocks for presentation purposes (see for example, 
           figure 5 in the paper "The K-SVD: An Algorithm for Designing of Overcomplete Dictionaries for Sparse Representation".
11. my_im2col - similar to the function 'im2col', only allow defining the sliding distance between the blocks.

The following 3 files implements the OMP (orthogonal matching pursuit) algorithm and the non-negative basis pursuit algorithm. 
This algorithm is used by the above KSVD and NN-KSVD functions. 
However, different sparse coding functions (or, implementations) may also be used by changing the relevant call in the KSVD file.
====================================================================================================================================

12. OMP - OMP algorithm. Finds a representation with fixed number of coefficients for each signal.
13. OMPerr - OMP algorithm. Finds a representation to the signals, allowing a (given) maximal representation error for each.
14. NN_BP - non-negative variation of the basis pursuit. finds a non-negative sparse representatation with a fixed number of coefficients for each signal.

For comments or questions please turn to Michal aharon (michal.aharon@hp.com) or Michael Elad (elad@cs.technion.ac.il).

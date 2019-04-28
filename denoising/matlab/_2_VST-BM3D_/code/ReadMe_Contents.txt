invansc package   v3.00  16.3.2016    [for Matlab v.7 (or later)]
-------------------------------------------------------------------
 Lucio Azzari, Markku Mäkitalo, Alessandro Foi


The Matlab software included in this package implements the transformations and filtering procedures for Poisson and Poisson-Gaussian data presented in the papers
[1] M. Mäkitalo and A. Foi, "On the inversion of the Anscombe transformation in low-count Poisson image denoising", Proc. Int. Workshop on Local and Non-Local Approx. in Image Process., LNLA 2009, Tuusula, Finland, pp. 26-32, August 2009. doi:10.1109/LNLA.2009.5278406
[2] M. Mäkitalo and A. Foi, "Optimal inversion of the Anscombe transformation in low-count Poisson image denoising", IEEE Trans. Image Process., vol. 20, no. 1, pp. 99-109, January 2011. doi:10.1109/TIP.2010.2056693
[3] M. Mäkitalo and A. Foi, "A closed-form approximation of the exact unbiased inverse of the Anscombe variance-stabilizing transformation", IEEE Trans. Image Process., vol. 20, no. 9, pp. 2697-2698, September 2011. doi:10.1109/TIP.2011.2121085
[4] M. Mäkitalo and A. Foi, "Optimal inversion of the generalized Anscombe transformation for Poisson-Gaussian noise", IEEE Trans. Image Process., vol. 22, no. 1, pp. 91-103, January 2013. doi:10.1109/TIP.2012.2202675
[5] L. Azzari and A. Foi, "Variance Stabilization for Noisy+Estimate Combination in Iterative Poisson Denoising", submitted, March 2016.


Software and publications can be downloaded from http://www.cs.tut.fi/~foi/invansc/

-------------------------------------------------------------------
 Contents
-------------------------------------------------------------------

demo_Poisson_experiments_table.m                      - script reproducing the results in Table 1 of the LNLA2009 paper [1].
demo_GenAnscombe_denoising.m                          - script reproducing the results in Table 1 of the paper [4]
demo_iterVSTpoisson.m                                 - script reproducing the results in Table 1 and Table 2 of paper [5].

Poisson_denoising_Anscombe_exact_unbiased_inverse.m   - function for denoising Poisson images using the Anscombe variance-stabilizing transformation, the BM3D filter, and the exact unbiased inverse of the Anscombe transformation
iterVSTpoisson.m                                      - function for denoising Poisson images using the iterative VST framework [5]

Anscombe_forward.m                                    - Anscombe variance-stabilizing transformation
GenAnscombe_forward.m                                 - generalized Anscombe variance-stabilizing transformation
Anscombe_inverse_asympt_unbiased.m                    - asymptotically unbiased inverse of the Anscombe transformation
Anscombe_inverse_exact_unbiased.m                     - exact unbiased inverse of the Anscombe transformation
GenAnscombe_inverse_exact_unbiased.m                  - exact unbiased inverse of the generalized Anscombe transformation
Anscombe_inverse_closed_form.m                        - closed-from approximation of exact unbiased inverse of the Anscombe transformation
GenAnscombe_inverse_closed_form.m                     - closed-from approximation of exact unbiased inverse of the generalized Anscombe transformation
Anscombe_inverse_MMSE.m                               - MMSE inverse of the Anscombe transformation
bin_B_h.m                                             - binning function
debin_Binv_h.m                                        - debinning function

Anscombe_vectors.mat                                  - precomputed expectation vectors used for defining the exact unbiased inverse of the Anscombe transformation
GenAnscombe_vectors.mat                               - precomputed expectation vectors and matrix used for defining the exact unbiased inverse of the generalized Anscombe transformation
MMSEcurves.mat                                        - precomputed curves used for defining the MMSE inverse of the Anscombe transformation
Anscombe_lambda.mat                                   - precomputed expectation vectors and matrix for defining the exact unbiased inverse of noisy+estimate combinations [5]
paramsFromQfun.mat                                    - function handle to set parameters in iterVSTpoisson.m [5]

./images (folder)                                     - test images used for the simulation experiments [2]
./images/images_readme.txt                            - information about the test images contained in images.mat and used for the simulation experiments
./images_for_table_1 (folder)                         - test images used for Table 1 of [5]
./images_for_table_2 (folder)                         - test images used for Table 2 of [5]

LEGAL_NOTICE.txt                                      - TUT license and disclaimers

ReadMe_Contents.txt                                   - this file


------------------------------------------------------------------
Installation and requirements
------------------------------------------------------------------
This software is designed to run on
*) MS Windows, Linux, or Mac OSX (32-bit or 64-bit CPU)
*) Matlab v.7 or later

The denoising functions are configured to use:
 * BM3D denoising filter
    http://www.cs.tut.fi/~foi/GCF-BM3D/

The demo scripts use also:
 * Statistics Toolbox (to generate Poisson data "poissrnd"),
 * Image Processing Toolbox (only for visualization with "imshow").


-------------------------------------------------------------------
 Change log
-------------------------------------------------------------------
v3     (16 March 2016)
 + files for iterative VST method [5]

v2.03  (10 August 2012)
 + bugfix (handling of default value of alpha=1)

v2.02  (9 August 2012)
 + bugfix (sigma normalized by alpha in the codes of the inverses)

v2.01  (2 October 2011)
 + updated some comments.

v2.00  (1 October 2011)
 + added routines for the Poisson-Gaussian case based on the generalized Anscombe transformation [4].
 + added closed-form approximation of exact unbiased inverse [2].

v1.02  (5 November 2009)
 + bugfix ('negative values' in Anscombe_inverse_asympt_unbiased.m).
 
v1.01  (5 November 2009)
 + added MMSE inverse.
 
v1.00  (9 September 2009) 
 * Initial release [1].

-------------------------------------------------------------------
 Disclaimer
-------------------------------------------------------------------

Any unauthorized use of these routines for industrial or profit-oriented activities is expressively prohibited. By downloading and/or using any of these files, you implicitly agree to all the terms of the TUT limited license, as specified in the document LEGAL_NOTICE.txt (included in this package) and online at http://www.cs.tut.fi/~foi/invansc/legal_notice.html


   Tampere University of Technology - 2011-2016
-----------------------------------------------------------------------------------

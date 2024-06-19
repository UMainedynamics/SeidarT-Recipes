Boundary Performance 
--------------------

Testing the performance of the CPML allows for building optimal models. There are 3 primary parameters - :math:`\kappa`, :math:`\sigma`, and :math:`\alpha` - for the absorbing boundary. 

:math:`\alpha` is an artificial loss parameter that add an additional loss term to stabilize the CPML. This is primarily effective in eliminating evancescent and grazing waves with higher max values enhancing the dampening effect. Typically, a small non-zero value is used, and while there are a few equations for computing an optimum :math:`\alpha_{\text{max}}`, it was found that

.. math:: 
	
	\alpha_{\text{max}} = 2 \pi f_{\text{src}} \varepsilon_0

provides a generalized term for numerical stability and optimal performance.  

**:math:`\kappa`** is the scaling factor that adjust the effective speed of the wave within the CPML. :math:`\kappa` enhances absorption by compressesing the wave field and increasing the path length. A higher :math:`\kappa_{\text{max}}` value leads to better absorption by compressing the fields more, but values that are too high can cause numerical instability. Lower values improve stability but at the cost of less effective boundaries. An optimal value is given by 

.. math::
	
	 \kappa_{i,\text{max}} = 1 + (\alpha_{\text{max}} / f_{\text{src}}) \sqrt{\mu_0 * \varepsilon_i}

however theory and practice sometimes diverge given the problem. When this happens, tuning the value involves a more heuristic approach.   

:math:`\sigma` represents the electrical conductivity within the CPML which introduces a loss term that attenuates the wave. This is the main damping mechanism in the CPML. Higher :math:`\sigma_{\text{max}}` values will increase damping however, excessively high values can cause rapid change in the wave field and numerical instability. The optimal sigma value can be computed using

.. math::
	
	\sigma_{i,\text{opt}} = \frac{NP + 1}{\Delta x_i \sqrt{\mu_0/(\varepsilon_i \varepsilon_0)} }

followed by a scaling factor, :math:`\rho`, 

.. math:: 	
	
	\sigma_{i,\text{max}} = \rho \cdot \sigma_{i,\text{opt}} 

There are different approaches to estimating the reflected power. This script computes the cumulative energy density in the boundary region over the full time interval by using a grid search over varying the proportionality constant, :math:`\rho`, and :math:`\kappa_{\text{max}}` values. Relative to a reference value, we can gauge the performance of the boundary conditions.



 


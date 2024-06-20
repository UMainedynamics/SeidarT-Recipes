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

followed by a scaling factor, :math:`\rho`, to find the maximum value

.. math:: 	
	
	\sigma_{i,\text{max}} = \rho \cdot \sigma_{i,\text{opt}} 

There are different approaches to estimating the reflected power. This script computes the cumulative energy density in the boundary region over the full time interval by using a grid search over varying the proportionality constant, :math:`\rho`, and :math:`\kappa_{\text{max}}` values. Relative to a reference value, we can gauge the performance of the boundary conditions. The parameters can be adjusted by the user to optimize them. They are defined in the *prjrun* module and consist of: *kappa_max*, *sig_opt_scalar*, *alpha_max_scalar*, *Rcoef*, *NP*, and *NPA*. The reflection coefficient, *Rcoef*, is specific to seismic CPML, and *NP* and *NPA are the exponential terms for grading the CPML from the inner boundary to the outer boundary max value. *NPA* is specific for calculating :math:`\alpha` since it is auxiliary to both :math:`\sigma` and :math:`\kappa`. If changing these parameters is desired, you can do so by importing the *prjrun* module and directly assigning the variables: For instance,

.. code-block:: python 
	
	from seidart.routines import prjrun 
	
	prjrun.kappa_max = 7e-1 
	prjrun.NP = 4 
	prjrun.NPA = 3
	prjrun.kappa_max = 1.1e0 
	prjrun.sig_opt_scalar = 0.8 
	prjrun.alpha_max_scalar = 1.1e0

The defualt values are:

::
	
	alpha_max_scalar = 1.0
	kappa_max = 5 
	sig_opt_scalar = 1.2
	*Rcoef* = 0.0010
	*NP* = 2
	*NPA* = 2


A breakdown of the code is as follows.

1. Import the appropriate modules.

.. code-block:: python
	
	import numpy as np
	from glob2 import glob
	import seaborn as sns

	from seidart.routines import prjrun, sourcefunction 
	from seidart.routines.definitions import * 


	prjfile = 'em_boundary_performance.prj'
	domain, material, __, em = prjrun.domain_initialization(prjfile)


2. Define the range of the variables that we want to use in the estimate.

.. code-block:: python
	
	dk = 0.1
	ki = 1.0 
	kf = 15
	kappa_max = np.arange(ki, kf+dk, dk)

	sk = 0.05 
	si = 0.1
	sf = 10.0
	sig_opt_scalar = np.arange(si,sf+sk, sk)

	source_frequency = np.array([
		2.0e7
	])

	ref_kappa = 1.0 
	ref_sigma = 1.0

	m = len(kappa_max)
	n = len(sig_opt_scalar)
	p = len(source_frequency)

3. Prep the model and pre-allocate variables. 

.. code-block::
	
	cumulative_power_density_x = np.zeros([m,n,p])
	cumulative_power_density_z = np.zeros([m,n,p])
	datx = np.zeros([domain.nx + 2 * domain.cpml, domain.nz + 2*domain.cpml])
	datz = datx.copy()

	# Load the project and get going
	prjrun.status_check(
		em, material, domain, prjfile, append_to_prjfile = True
	)
	timevec, fx, fy, fz, srcfn = sourcefunction(em, 10, 'gaus1')

4. Loop through each value and 
	a. Overwrite the current values for *kappa_max* and *sig_opt_scalar*
	b. Run the model 
	c. Zero out the model domain within the boundaries in order to easily sum the energy.
	d. Re the domain dimensions to their original values. 

.. code-block:: python

	for ii in range(m):
		prjrun.kappa_max = kappa_max[ii]
		for jj in range(n):
			prjrun.sig_opt_scalar = sig_opt_scalar[jj] 
			for kk in range(p):
				print( f"{ii}/{m-1} - {jj}/{n-1} - {kk}/{p-1}" )
				# Zero out the 
				datx[:,:] = 0.0
				datz[:,:] = 0.0 
				em.f0 = source_frequency[kk]
				prjrun.runelectromag(
					em, material, domain
				)
				domain.nx = domain.nx + 2*domain.cpml
				domain.nz = domain.nz + 2*domain.cpml
				
				fnx = glob('Ex.*.dat')
				fnz = glob('Ez.*.dat')
				for filex, filez in zip(fnx, fnz):
					datx += ( read_dat(
						filex, 'Ex', domain, is_complex = False, single = True
					) )**2
					datz += ( read_dat(
						filez, 'Ez', domain, is_complex = False, single = True
					) )**2
				
				datx[domain.cpml:-(domain.cpml+1),domain.cpml:-(domain.cpml+1)] = 0.0
				datz[domain.cpml:-(domain.cpml+1),domain.cpml:-(domain.cpml+1)] = 0.0
				
				cumulative_power_density_x[ii,jj,kk] = datx.sum() 
				cumulative_power_density_z[ii,jj,kk] = datz.sum() 
				
				# Typically the domain dimensions are changed and stored in an Array
				# object, but in this case we have to manually change them so that 
				# an error isn't returned in runelectromag or read_dat
				domain.nx = domain.nx - 2*domain.cpml
				domain.nz = domain.nz - 2*domain.cpml

5. Visualize the output using a bivariate KDE plot. 

.. code-block:: python
	
	# This is setup to compute many different source frequencies. 
	cpd_x = cumulative_power_density_x[:,:,0]
	cpd_z = cumulative_power_density_z[:,:,0]

	# Create the kde plots 
	kappa_grid, sigma_grid = np.meshgrid(kappa_max, sig_opt_scalar)
	kappa_flat = kappa_grid.ravel() 
	sigma_flat = sigma_grid.ravel() 
	cpd_x_flat = cpd_x.ravel()
	cpd_z_flat = cpd_z.ravel()

	fig, axs = plt.subplots(1,2, figsize = (12,6), sharey = True)
	sns.kdeplot(x = sigma_flat, y = kappa_flat, weights = cpd_x_flat, fill = True, ax = axs[0], cmap = "Blues")
	sns.kdeplot(x = sigma_flat, y = kappa_flat, weights = cpd_z_flat, fill = True, ax = axs[1], cmap = "Blues")

	axs[0].set_xlabel(r'$\kappa_{\text{max}}$', fontsize = 16)
	axs[0].set_ylabel(r'$\sigma_{\text{max}}/\sigma_{\text{opt}}$', fontsize = 16)
	axs[0].set_title('Ex Cumulative Power Density')
	axs[1].set_xlabel(r'$\kappa_{\text{max}}$', fontsize = 16)
	axs[1].set_ylabel(r'$\sigma_{\text{max}}/\sigma_{opt}$', fontsize = 16)
	axs[1].set_title('Ez Cumulative Power Density')
	plt.show()

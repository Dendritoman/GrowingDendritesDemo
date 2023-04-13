#!/usr/bin/python3

# dnn_demo.py - Dendrite neural network demonstration
# Copyright (c) 2022-2023 Robert Baxter  (MIT License)

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

# This Python program requires Python 3 and the numpy library
# To run type: python3 dnn_demo.py

# For minimal printing to stdout, set verbose = 0
# If num_epochs > 5, suggest directing the output to a file, e.g.
#   python3 dnn_demo.py > dnn_demo_out.txt

# Example parameters for fast learning of the XOR patterns
#   epsilon_w  = 0.9
#   epsilon_g  = 0.999
#   num_epochs = 5 (gets 0 errors with num_epochs = 3 but weights haven't matured)
# Example parameters for slow learning of the XOR patterns
#   epsilon_w  = 0.1
#   theta_g    = 0.05
#   num_epochs = 50

# Note: This program is written for simplicity and clarity not for efficiency or production runs.
#       The computations can be ordered differently for efficiency, and the dendritic excitations,
#       neuron outputs, and error signals can be computed after synaptogenesis and after
#       synaptic modification for faster convergence.

# Imports
import sys, os
import numpy as np  # requires numpy library

# Parameters:
verbose            = 0      # 0=quiet during training,n=training progress every n epochs
epsilon_w          = 0.9    # synaptic modification rate parameter
epsilon_g          = 0.999  # synaptogenesis decrement rate parameter (decrement rate = 1 - epsilon_g)
alpha              = 1.0    # averaging rate of false negative (missed detection) error signal
gamma_init         = 1.0    # initial value of the synaptogenesis rate parameter
W_0                = 0.1    # initial value of synaptic weights
theta_w            = 0.005  # synaptic shedding threshold
theta_g            = 0.5    # dendritogenesis threshold
theta_md           = 0.05   # false negative (missed detection) error threshold
num_dendrites_init = 0      # initial number of dendrites on each neuron
num_dendrites_max  = 8      # maximum number of dendrites allowed on each neuron
num_neurons        = 2      # number of neurons (= number of distinct class labels)
num_epochs         = 5      # number of loops over the input patterns
rng_seed           = 124    # (psuedo)random number generator seed

# Input vectors (complement-coded XOR)
Xin = np.array([[0,0,1,1],[1,1,0,0],[0,1,1,0],[1,0,0,1]])
num_inputs   = len(Xin[0,:])  # number of input lines = dimensionality of input vectors
num_patterns = len(Xin[:,0])  # number of input patterns per epoch

# Class labels for each input vector (labels must be in [0,1,...,num_neurons-1])
Kin = np.array([0,0,1,1],dtype=int) # class labels for each input pattern
num_classes = len(np.unique(Xin,return_counts=True)[1]) # number of classes

# Check parameters
if num_dendrites_init < 0 or num_dendrites_init > num_dendrites_max:
	# The initial number of dendrites must be in the closed interval [0, num_dendrites_max]
	sys.exit("num_dendrites_init = %d but must be 0, 1, ..., or %d" % (num_dendrites_init,num_dendrites_max))
if num_neurons != num_classes:
	# In this demo, each class is a associated with a single neuron
	sys.exit("num_neurons = %d, num_classes = %d, they must be equal" % (num_neurons,num_classes))
for k in Kin:
	# In this demo, the class indices must match the neuron indices
	if Kin[k] not in range(num_neurons):
		sys.exit("Kin[%d] = %d, which does not match any neuron index" % (k,Kin[k]))

# Initialization
i_list = range(num_inputs)                  # list of input line indices
p_list = range(num_patterns)                # list of input pattern indices
j_list = range(num_neurons)                 # list of neuron indices
d_list = np.empty(num_neurons,dtype=object) # list of dendrite indices for each neuron
for j in j_list:
	d_list[j] = list(range(num_dendrites_init))
c_dji  = np.zeros((num_dendrites_max,num_neurons,num_inputs),dtype=int) # connection indicators
w_dji  = np.zeros((num_dendrites_max,num_neurons,num_inputs))           # synaptic weights
y_dj   = np.zeros((num_dendrites_max,num_neurons))                      # dendritic excitations
g_dj   = gamma_init*np.ones((num_dendrites_max,num_neurons))            # gamma_dj
dmax_j = np.zeros(num_neurons,dtype=int)                                # dendrite with max(y_dj) on j
y_j    = np.zeros(num_neurons)                                          # neuron excitations
z_j    = np.zeros(num_neurons,dtype=int)                                # neuron outputs
zs_j   = np.zeros(num_neurons,dtype=int)                                # supervision signals
fn_j   = np.zeros(num_neurons,dtype=int)                                # false negative (missed detection) error
fp_j   = np.zeros(num_neurons,dtype=int)                                # false positive (false alarm) error
fnra_j = np.ones(num_neurons)  # running average of false negative (missed detection) error signal
nshd_j = np.zeros(num_neurons) # number of shed connections
nnew_j = np.zeros(num_neurons) # number of new connections
EX     = np.zeros(num_inputs)  # E[X_i] = mean value of each input line
np.random.seed(rng_seed)       # initialize random number generator with rng_seed

# Pre-compute E[X_i]
# Note: Using E[X_i] in the synaptic modification rule (dw) is not required
for i in i_list:
	EX[i] = np.mean(Xin[:,i])

# Begin training
num_trials = 0 # number of pattern presentations
for epoch in range(num_epochs):
	p_list_shuffled = np.random.permutation(p_list) # shuffle presentation order of patterns
	for p in p_list_shuffled:
		num_trials = num_trials + 1 # number of training trials counter

		# Compute dendritic excitations for this input pattern
		buf_dengen = "" # string buffer used for printing status
		for j in j_list:
			dmax_j[j] = -1         # stores dendrite on neuron j with the maximum excitation
			ydj_max = -1.0         # stores maximum excitation for neuron j
			nd_j = len(d_list[j])  # current number of dendrites on neuron j
			if nd_j == 0:
				d_list[j] = [0] # if starting with no dendrites, add the first dendrite
				if len(buf_dengen) == 0:
					buf_dengen = "  added dendrite to j %d" % j
				else:
					buf_dengen = buf_dengen + ", %d" % j
			# Compute dendrite excitations, y_dj
			for d in d_list[j]:
				nc_dj = np.sum( c_dji[d,j,:] )  # number of connections to dendrite d
				if nc_dj > 0:
					# Compute the excitation for dendrite d on neuron j, y_dj
					sumw = np.sum(w_dji[d,j,:])
					y_dj[d,j] = np.dot(Xin[p,:],w_dji[d,j,:])/sumw
				else:
					# This dendrite has no connections so its excitation is 0
					y_dj[d,j] = 0.0
				if y_dj[d,j] > ydj_max:
					ydj_max = y_dj[d,j]
					if ydj_max > 0.0:
						dmax_j[j] = d
			y_j[j] = ydj_max  # maximum excitation for neuron j

		# Determine the neuron and dendrite with the maximum excitation
		yj_max = np.max(y_j) # maximum excitation across all neurons
		j_yj_max = np.argwhere( y_j == yj_max)[:,0] # neuron with the maximum excitation
		if yj_max == 0.0:
			# Maximum excitation is 0 so there is no winning neuron or dendrite
			j_max = -1
			d_max = -1
		else:
			if len(j_yj_max) > 1:
				# Multiple neurons have the maximum excitation; choose one
				j2_yj_max = np.random.permutation(j_yj_max)
				j_max = j2_yj_max[0]
				j_max = -1
				d_max = -1
			else:
				# Find the neuron with the maximum excitation
				j_max = np.argmax(y_j)
			d_max = np.argmax( y_dj[:,j_max] ) # dendrite with the maximum excitation

		# Determine the neuron outputs
		z_j[:] = 0
		if j_max >= 0:
			z_j[j_max] = 1

		# Determine the error signals
		for j in j_list:
			zs_j[j] = j == Kin[p] # supervision signal
			fn_j[j] = j == Kin[p] and z_j[j] == 0  # false negative (missed detection) error signal
			fp_j[j] = j != Kin[p] and z_j[j] == 1  # false positive (false alarm) error signal
			fnra_j[j] = (1.0-alpha)*fnra_j[j] + alpha*fn_j[j] # update false negative running average

		# Check for success
		if j_max == Kin[p]:
			# Success: decrement gamma on dendrite d_max on neuron j_max
			g_dj[d_max,j_max] = g_dj[d_max,j_max] * (1.0 - epsilon_g)

		# Check for dendritogenesis
		for j in j_list:
			d_last = d_list[j][-1]
			if g_dj[d_last,j] < theta_g and fnra_j[j] > theta_md:
				# Add a dendrite to this neuron
				if len(d_list[j]) < num_dendrites_max:
					d_list[j] = d_list[j] + [d_last+1]
					if len(buf_dengen) == 0:
						buf_dengen = "  added dendrite to j %d" % j
					else:
						buf_dengen = buf_dengen + ", %d" % j

		# Perform synaptogenesis
		for j in j_list:
			if j == Kin[p]:
				for d in d_list[j]:
					p_dj = g_dj[d,j]*fn_j[j]*Xin[p,:]
					rn = np.random.rand(num_inputs)
					new_connections = p_dj > rn
					new_connections = new_connections*(1-c_dji[d,j,:]) # ignore existing connections
					num_new = np.count_nonzero(new_connections)
					if num_new > 0:
						c_dji[d,j,:] = c_dji[d,j,:] + new_connections
						w_dji[d,j,:] = w_dji[d,j,:] + W_0 * new_connections

		# Update synaptic weights
		if d_max >= 0:
			# Update synaptic weights only on dendrite dmax_j on each neuron
			for j in j_list:
				d = dmax_j[j]
				dw = c_dji[d,j,:]*epsilon_w * (Xin[p,:] - EX - w_dji[d,j,:]) * y_dj[d,j] * zs_j[j]
				w_dji[d,j,:] = w_dji[d,j,:] + dw

		# Shed synaptic weights
		for j in j_list:
			for d in d_list[j]:
				nc_dj = np.sum( c_dji[d,j,:] )
				if nc_dj > 0:
					for i in i_list:
						if w_dji[d,j,i] < theta_w and c_dji[d,j,i] == 1:
							w_dji[d,j,i] = 0.0
							c_dji[d,j,i] = 0

		if verbose > 0 and ((epoch+1) % verbose) == 0: #or epoch == (num_epochs-1):
			# Display status
			buf = "epoch %3d p %d k %d z_j %d %d fn %d %d fp %d %d dmax_j %2d %2d d_max %2d"\
					% (epoch+1,p,Kin[p],z_j[0],z_j[1],fn_j[0],fn_j[1],fp_j[0],fp_j[1],\
					   dmax_j[0],dmax_j[1],d_max)
			for j in j_list:
				buf = buf + " nc_%d:" % j
				for d in d_list[j]:
					nc_dj = np.sum( c_dji[d,j,:] )
					buf = buf + " %d" % nc_dj
			sys.stdout.write(buf + "\n")
			print("      Xi= %4d %4d %4d %4d  k=%d" % (Xin[p,0],Xin[p,1],Xin[p,2],Xin[p,3],Kin[p]))
			buf = ""
			for j in j_list:
				for d in d_list[j]:
					buf = "      w_%d%d= %4.2f %4.2f %4.2f %4.2f"\
						  % (d,j,w_dji[d,j,0],w_dji[d,j,1],w_dji[d,j,2],w_dji[d,j,3])
					buf = buf + " y_%d%d=%5.2f" % (d,j,y_dj[d,j])
					buf = buf + " g_%d%d=%6.4f" % (d,j,g_dj[d,j])
					print(buf)
			if len(buf_dengen) > 0:
				print(buf_dengen)
# End training

print("\nCompleted %d training epochs (%d patterns per epoch, %d trials)" % (num_epochs,num_patterns,num_trials))

# Begin testing
num_errors = 0
for p in p_list:
	# Compute dendritic excitations for this input pattern
	for j in j_list:
		dmax_j[j] = -1
		ydj_max = -1.0
		for d in d_list[j]:
			nc_dj = np.sum( c_dji[d,j,:] )
			if nc_dj > 0:
				sumw = np.sum(w_dji[d,j,:])
				y_dj[d,j] = np.dot(Xin[p,:],w_dji[d,j,:])/sumw
			else:
				y_dj[d,j] = 0.0
			if y_dj[d,j] > ydj_max:
				ydj_max = y_dj[d,j]
				if ydj_max > 0.0:
					dmax_j[j] = d
		y_j[j] = ydj_max  # maximum excitation for this neuron

	# Determine the neuron and dendrite with the maximum excitation
	yj_max = np.max(y_j) # maximum excitation across all neurons
	j_yj_max = np.argwhere( y_j == yj_max )[:,0] # neuron with the maximum excitation
	if yj_max == 0.0:
		# Maximum excitation is 0 so there is no winning dendrite
		j_max = -1
		d_max = -1
	else:
		if len(j_yj_max) > 1:
			# Multiple neurons have the maximum excitation; choose one
			j2_yj_max = np.random.permutation(j_yj_max)
			j_max = j2_yj_max[0]
		else:
			# Find the neuron and dendrite with the maximum excitation
			j_max = np.argmax(y_j)
		d_max = np.argmax( y_dj[:,j_max] )

	# Determine the neuron outputs
	z_j[:] = 0
	if j_max >= 0:
		z_j[j_max] = 1

	if j_max != Kin[p]:
		num_errors = num_errors + 1

	# Determine the error signals
	for j in j_list:
		zs_j[j] = j == Kin[p] # supervision signal
		fn_j[j] = j == Kin[p] and z_j[j] == 0  # false negative (missed detection) error signal
		fp_j[j] = j != Kin[p] and z_j[j] == 1  # false positive (false alarm) error signal

	# Display test results
	print("p %d:    Xi=%4d %4d %4d %4d k=%d z_j %d %d d_max %d"\
	         % (p,Xin[p,0],Xin[p,1],Xin[p,2],Xin[p,3],Kin[p],z_j[0],z_j[1],d_max))
	if d_max >= 0:
		print("      w_dj= %4.2f %4.2f %4.2f %4.2f"\
		         % (w_dji[d_max,j_max,0],w_dji[d_max,j_max,1],w_dji[d_max,j_max,2],w_dji[d_max,j_max,3]))
# End testing

print("Number of test errors = %d out of %d patterns" % (num_errors,num_patterns))

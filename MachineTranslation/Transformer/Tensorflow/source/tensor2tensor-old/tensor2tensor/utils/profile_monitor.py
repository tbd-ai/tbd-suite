# <EcoSys> Added this file.

import numba.cuda as cuda
import tensorflow as tf

class ProfileMonitor(tf.contrib.learn.monitors.EveryN):
	def __init__(self):
		super(ProfileMonitor, self).__init__(5)
		self.started = False
		self.ended = False

	def every_n_step_begin(self, step):

		if self.ended:
			return		

		first_check_step = 305
		last_check_step = 325
		if (not self.started) and step > first_check_step:
			print("Profile Start!")
			self.started = True
			cuda.profile_start()
		elif self.started and step > last_check_step:
			print("Profile End! Calling profile_stop().")
			self.ended = True
			cuda.profile_stop()
			print("Done calling profile_stop().")

# </EcoSys>

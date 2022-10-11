from signature.utils import distributions
from signature.models import OU, MA2, GSE

def get_task(task):

	try:
		task_var = eval(task)
	except:
		raise NotImplemented("Task not recognised")
	prior = task_var.prior
	sbi_prior = task_var.torch_prior
	model = task_var.Model()
	obs  = task_var.data

	def simulator(theta):
		data = model.simulate(pars=theta.tolist())
		return data

	return prior, sbi_prior, obs, simulator

'''
This file is used to allow the skrf vi docs to be build without having 
a working visa module 
'''

class Dummy(object):
	def __init__(self, *args, **kwargs):
		pass 
	
	def __getattr__(self,name):
		print 'tried to get %s'%name
		return Dummy1()

class Dummy1(object):
	def __init__(self, *args, **kwargs):
		pass 
	
	def __getattr__(self,name):
		print 'tried to get %s'%name
		return Dummy2()
	
	def __call__(self, *args, **kwargs):
		return Dummy2()
		
class Dummy2(object):
	def __init__(self, *args, **kwargs):
		pass 
	
	def __getattr__(self,name):
		print 'tried to get %s'%name
		return Dummy3()
	
	def __call__(self, *args, **kwargs):
		return Dummy3()
		
class Dummy3(object):
	def __init__(self, *args, **kwargs):
		pass 
	
	def __getattr__(self,name):
		print 'tried to get %s'%name
		return 1

		
class GpibInstrument(Dummy):
	def __init__(self, *args, **kwargs):
		pass 
	
	def write(self,msg, *args, **kwargs):
		'''
		dummy doc
		'''
		print msg
		pass
	
	

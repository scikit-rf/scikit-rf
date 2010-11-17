from Tkinter import *
from tkFileDialog import askopenfilename


class App:
	def __init__(self,master):
		self.numberofrows = 0
		self.filenamedict = {}
		
		self.mainframe = Frame(root)
		self.mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
		self.mainframe.columnconfigure(0, weight=1)
		self.mainframe.rowconfigure(0, weight=1)

		Button(self.mainframe, text="Add Line", command=self.addline).grid(column=2, row=1, sticky=W)
		for child in self.mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)
	
	def addline(self,*args):
		self.numberofrows+=1
		row = self.numberofrows +1
		self.filenamedict[str(self.numberofrows )] = StringVar()
		filename = self.filenamedict[str(self.numberofrows )]
		Label(self.mainframe, text = 'std #'+str(self.numberofrows)).grid(column=1, row=row , sticky=(W, E))
		Entry(self.mainframe, width=20, textvariable=filename).grid(column=2, row=row , sticky=(W, E))
		Button(self.mainframe, text="Open", command=self.openfile).grid(column=3, row=row  , sticky=W)
		Label(self.mainframe, textvariable=filename).grid(column=4, row=row , sticky=(W, E))

	def openfile(self,*args):
		self.filenamedict[str(self.numberofrows)].set(askopenfilename())	




root = Tk()
app = App(root)
root.mainloop()


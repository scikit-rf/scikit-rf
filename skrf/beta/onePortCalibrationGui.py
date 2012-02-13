from Tkinter import *
from tkFileDialog import askopenfilename


class App:
    def __init__(self,master):
        self.numberofstandards = 0
        self.filenamedict = {}

        self.mainframe = Frame(root)
        self.mainframe.grid(column=2, row=3, sticky=(N, W, E, S))
        self.mainframe.columnconfigure(0, weight=1)
        self.mainframe.rowconfigure(0, weight=1)

        Button(self.mainframe, text="Add Standard", command=self.addstandard).grid(column=2, row=1, sticky=W)
        Button(self.mainframe, text="Delete Standard").grid(column=3, row=1, sticky=W)
        Button(self.mainframe, text="Calibrate").grid(column=4, row=1, sticky=W)
        Label(self.mainframe, text = 'Ideals').grid(column=2, row=2 , sticky=(W, E))
        Label(self.mainframe, text = 'Measurements').grid(column=4, row=2 , sticky=(W, E))
        for child in self.mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)

    def addstandard(self,*args):
        self.numberofstandards+=1
        row = self.numberofstandards +2
        self.filenamedict[str(self.numberofstandards )] = StringVar()
        filename = self.filenamedict[str(self.numberofstandards )]
        Label(self.mainframe, text = 'std #'+str(self.numberofstandards)).grid(column=1, row=row , sticky=(W, E))
        Entry(self.mainframe, width=20, textvariable=filename).grid(column=2, row=row , sticky=(W, E))
        Button(self.mainframe, text="Open", command=self.openfile).grid(column=3, row=row  , sticky=W)
        Entry(self.mainframe, width=20, textvariable=filename).grid(column=4, row=row , sticky=(W, E))
        Button(self.mainframe, text="Open", command=self.openfile).grid(column=5, row=row  , sticky=W)
        #Label(self.mainframe, textvariable=filename).grid(column=4, row=row , sticky=(W, E))

    def openfile(self,*args):
        self.filenamedict[str(self.numberofstandards)].set(askopenfilename())




#root = Tk()
#app = App(root)
#root.mainloop()

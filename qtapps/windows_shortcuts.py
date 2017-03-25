from win32com.client import Dispatch


def createShortcut(path, target='', wDir='', args= '', icon=''):
    shell = Dispatch('WScript.Shell')
    shortcut = shell.CreateShortCut(path)
    shortcut.TargetPath = target
    shortcut.WorkingDirectory = wDir

    if args:
        shortcut.Arguments = args

    if icon == '':
        pass
    else:
        shortcut.IconLocation = icon
    shortcut.save()

path = r"C:\Users\A3R7LZZ\Desktop\Multiline TRL.lnk"
target = r"C:\Miniconda3\pythonw.exe"
args = r"C:\Coding\Python\scikit-rf\qtapps\multiline_trl.py"
createShortcut(path, target, args=args)


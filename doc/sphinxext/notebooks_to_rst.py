
import os, shutil, string, glob, fnmatch,re
from IPython.nbconvert.writers import FilesWriter
from IPython.nbconvert.exporters.rst import RSTExporter
from IPython.nbconvert.nbconvertapp import NbConvertApp
from runipy.notebook_runner import NotebookRunner

import logging
log_format = '%(asctime)s %(message)s'
log_datefmt = '%m/%d/%Y %I:%M:%S %p'



def recursive_find_by_filter (dir, filter_str, include_hidden=False ):
    '''
    recursively search for files matching filter_str, starting at `dir`
    
    taken from
    http://stackoverflow.com/questions/2186525/use-a-glob-to-find-files-recursively-in-python
    '''
    matches = []
    for root, dirnames, filenames in os.walk(dir):
        if not include_hidden:
            filenames = [f for f in filenames if not f[0] == '.']
            dirnames[:] = [d for d in dirnames if not d[0] == '.']
        for filename in fnmatch.filter(filenames, '%s'%filter_str):
            matches.append(os.path.join(root, filename))
    return matches

def find_replace(filename, pattern, replace):
    '''
    sed-like find-replace function
    
    taken from  
    http://stackoverflow.com/questions/4427542/how-to-do-sed-like-text-replace-in-python)
    '''
    with open(filename, "r") as sources:
        lines = sources.readlines()
    with open(filename, "w") as sources:
        for line in lines:
            sources.write(re.sub(pattern, replace, line))
    
    

def go(source_dir):
    print os.listdir(source_dir)
    if not os.path.exists(source_dir):
        raise ValueError ('source_dir doesnt exist')
        
    print '----- starting NB Evaluation and Conversion'
    logging.basicConfig(level=logging.DEBUG, format=log_format, datefmt=log_datefmt)
    nb_files = recursive_find_by_filter(source_dir, '*.ipynb')
    print 'source dir is  %s'%source_dir
    for nb_file in nb_files:
        print nb_file
        basename = os.path.basename(nb_file)
        notebook_name = basename[:basename.rfind('.')]
        build_directory = os.path.dirname(nb_file)
        
        r = NotebookRunner(nb_file, pylab=True)
        r.run_notebook()
        r.save_notebook(nb_file)
        
        
        exporter = RSTExporter()
        writer =  FilesWriter()
        
        resources={}
        resources['output_files_dir'] = '%s_files' % notebook_name
        
        output, resources = exporter.from_notebook_node(r.nb, 
                                                resources = resources)
        
        
        writer.build_directory = build_directory
        writer.write(output, resources, notebook_name=notebook_name)
        rst_file = nb_file[:nb_file.rfind('.')]+'.'+exporter.file_extension
        
        # this could be improved to only change the double quotes from
        # cross references (ie  :.*:``.*`` -> :.*:`.*`)
        find_replace(rst_file,r'``','`')
        

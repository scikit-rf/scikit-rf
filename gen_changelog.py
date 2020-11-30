#!/usr/bin/env python
'''Script to build a changelog for all version tags'''
from subprocess import Popen, PIPE, CalledProcessError, check_call
import datetime

## Variables
changelog_filename = 'CHANGELOG.txt'
header = '''Changelog For scikit-rf
Generated on %s


'''%(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))



## Functions
def sh(cmd):
    """Execute command in a subshell, return status code."""
    return check_call(cmd, shell=True)

def sh2(cmd, ignore_retcode = False):
    """Execute command in a subshell, return stdout.

    Stderr is unbuffered from the subshell.x"""
    p = Popen(cmd, stdout=PIPE, shell=True)
    out = p.communicate()[0]
    retcode = p.returncode
    if ignore_retcode:
        return out.rstrip()
    else:
        if retcode:
            raise CalledProcessError(retcode, cmd)
        else:
            return out.rstrip()

## Start
with open(changelog_filename,'w') as fid:
    fid.write(header)
    tags = sh2('git tag').split('\n')
    
    
    for k in range(len(tags))[1::-1]:
        fid.write('\n\n-------------------------------- %s ---------------------------------\n\n'%tags[k+1])
        fid.write(sh2('git shortlog -n %s..%s'%(tags[k],tags[k+1]), ignore_retcode=True))
    
    fid.write ('\n----------------------------------- %s ---------------------------------\n\n'%tags[0])
    fid.write(sh2('git shortlog -n %s'%(tags[0]), ignore_retcode=True))


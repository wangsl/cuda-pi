#!/bin/bash

# $Id$

git add *.[FChm] *.cu *.sh *.pl Makefile

#svnId *.[FChm] *.pl *.sh *.inc Makefile

exit

git commit -m "comments"

git push origin master

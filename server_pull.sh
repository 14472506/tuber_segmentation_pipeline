#!/bin/bash
# This script will update pull when using in places like colab. Script will destory any changes!
#
#
#

echo "Begginging Pull"
# fetching all remot repos
git fetch --all
# git branching to make backup master
git branch backup-master
# !throwing away all change do not develope then call this
git reset --hard origin
# pulling from git
git pull
echo "Pull complete"
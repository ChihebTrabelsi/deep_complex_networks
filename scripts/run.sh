#!/usr/bin/env bash
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export PYTHONUNBUFFERED=1
echo "$USER running on $HOSTNAME"
exec $@

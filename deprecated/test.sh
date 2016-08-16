#!/usr/bin/expect -f

"""
how to run:
go to the folder where the .sh lies,
run /usr/bin/expect test.sh
"""
# connect via scp
spawn scp ejlq@da74wbedge1:/home/ejlq/RoofEst.py /Users/ejlq/Documents

#######################
expect {
  -re ".*es.*o.*" {
    exp_send "yes\r"
    exp_continue
  }
  -re ".*sword.*" {
    exp_send "tbN1dcsgium\r"
  }
}
interact
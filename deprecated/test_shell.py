import os
os.system("cat /root/test.sh")
#!/bin/bash
x='1'
while [[ $x -le 10 ]] ; do
    echo $x: hello $1 $2 $3
    sleep 1
    x=$(( $x + 1 ))
done

arglist = 'arg1 arg2 arg3'
bashCommand = 'bash /root/test.sh ' + arglist
os.system(bashCommand)
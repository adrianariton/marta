#!/bin/bash
PORT=${1:-8888}
ssh -N -o StrictHostKeyChecking=no -R $PORT:localhost:$PORT fep &
TUNNEL_PID=$!

FEP="<fep>"
echo "======================================"
echo "On your local machine:"
echo "Make the ssh tunnel as such: ssh -L $PORT:localhost:$PORT $USER@$FEP"
echo "Or: ssh -L $PORT:localhost:$PORT fep"
echo "Assuming you have fep in your known hosts, which you can add like so:"
echo "Host fep"
echo "    HostName fep.cluster.edu"
echo "    User your-username"
echo "======================================"
echo "Then access the seaker server at http://localhost:$PORT/"
echo "======================================"
echo "Common errors:"
echo "  - '[Errno 98] error while attempting to bind on address ('0.0.0.0', $PORT): address already in use'. Fix: kill the slurm job and run 'fuser -k $PORT/tcp' on the fep server"
echo "  - 'ssh_askpass: exec(/usr/libexec/openssh/ssh-askpass): No such file or directory'. Fix: kill the slurm job and run 'cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys' on fep"
echo "Testing: Run 'wget http://localhost:8888/api/health' on your local machine."
apptainer exec --nv env.sif python server.py --port $PORT

#!/bin/bash

# Run both visualize.sh commands (first in background)
./visualize.sh -b datasets/crescendo_sorry_qwen3n_vs_cbadicbmixed2t001__async_batch_bagel_eval_3repl_3x8x5 -t crescendo -p 5001 & \
./visualize.sh -b datasets/crescendo_sorry_qwen3n_vs_cbllama__async_batch_bagel_eval_3repl_3x8x5 -t crescendo & \
./visualize.sh -b ./datasets/fitd_hb_qwen3n_vs_dertains__async_batch_bagel_eval_3repl -t fitd -p 5002 &
# Open host.html in the default browser
xdg-open host.html 2>/dev/null || open host.html 2>/dev/null || echo "Could not auto-open host.html"

# SSH tunnel (foreground — keeps script alive)
ssh -L 8004:localhost:8004 fep
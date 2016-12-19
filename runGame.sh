#!/bin/bash

./halite -t -d "30 30" "TF_CPP_MIN_LOG_LEVEL=3 python MyBot.py" "TF_CPP_MIN_LOG_LEVEL=3 python halite-match-manager/bots/OverkillBot/MyBot.py"
#./halite -t -d "30 30" "TF_CPP_MIN_LOG_LEVEL=3 python history/dist-3/MyBot.py" "TF_CPP_MIN_LOG_LEVEL=3 python OverkillBot/OverkillBot.py"
#./halite -t -d "30 30" "TF_CPP_MIN_LOG_LEVEL=3 python history/dist-3/MyBot.py" "TF_CPP_MIN_LOG_LEVEL=3 python history/dist-1/MyBot.py"

#!/usr/bin/env bash
set -e

# Nuke any bad env that Streamlit might read
unset STREAMLIT_SERVER_PORT
unset STREAMLIT_SERVER_ADDRESS

# Hard-code 8080 (you said you have it open)
exec streamlit run main.py \


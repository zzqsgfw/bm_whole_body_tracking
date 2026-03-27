#!/usr/bin/env bash

# Source this file before launching wandb-backed training.
# Example:
#   source scripts/wandb_env.sh
#   export WANDB_API_KEY=...
#   wandb login --relogin

if [ -n "${WANDB_ENTITY:-}" ] && [ -z "${WANDB_USERNAME:-}" ]; then
  export WANDB_USERNAME="$WANDB_ENTITY"
fi
export WANDB_PROJECT="${WANDB_PROJECT:-beyondmimic-tracking}"

echo "WANDB_ENTITY=${WANDB_ENTITY:-<unset>}"
echo "WANDB_USERNAME=${WANDB_USERNAME:-<unset>}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"

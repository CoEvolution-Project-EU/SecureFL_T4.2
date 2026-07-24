#!/bin/bash

# ATTACKS=("Semantic-Label-Flip" "Gaussian" "Sign-Flip" "IPM" "ALIE" "None")
ATTACKS=("Sign-Flip" "IPM" "Gaussian" "ALIE" "None")
DEFENCES=("FedCluster" "FedGreed" "Mean" "Multi-Krum" "RFA" "FL-Defender" "FoolsGold")

NUM_CLIENTS=20
NUM_MALICIOUS_CLIENTS=10

echo "Setting fixed config values: num_clients=${NUM_CLIENTS}, num_malicious_clients=${NUM_MALICIOUS_CLIENTS}"

# Set num_clients safely (scoped under client:)
awk -v nc="${NUM_CLIENTS}" '
  /^client:/ { in_client = 1 }
  /^[^ ]/ && !/^client:/ { in_client = 0 }
  in_client && /^  num_clients:/ { $0 = "  num_clients: " nc }
  { print }
' config.yaml > config.yaml.tmp && mv config.yaml.tmp config.yaml

# Set num_malicious_clients safely (scoped under attack:)
awk -v nmc="${NUM_MALICIOUS_CLIENTS}" '
  /^attack:/ { in_attack = 1 }
  /^[^ ]/ && !/^attack:/ { in_attack = 0 }
  in_attack && /^  num_malicious_clients:/ { $0 = "  num_malicious_clients: " nmc }
  { print }
' config.yaml > config.yaml.tmp && mv config.yaml.tmp config.yaml

for ATTACK in "${ATTACKS[@]}"; do
  for DEFENCE in "${DEFENCES[@]}"; do
    echo "=================================================="
    echo "Starting simulation for Attack: \"$ATTACK\" | Defence: \"$DEFENCE\""
    echo "=================================================="
    
    # # Skip FedCluster when using Sign-Flip
    # if [ "$ATTACK" = "Sign-Flip" ] && [ "$DEFENCE" = "FedCluster" ]; then
    #   echo "Skipping combination Attack: $ATTACK | Defence: $DEFENCE"
    #   echo ""
    #   continue
    # fi
    
    # Update Attack type safely (scoped under attack:)
    awk -v att="${ATTACK}" '
      /^attack:/ { in_attack = 1 }
      /^[^ ]/ && !/^attack:/ { in_attack = 0 }
      in_attack && /^  type:/ { $0 = "  type: " att }
      { print }
    ' config.yaml > config.yaml.tmp && mv config.yaml.tmp config.yaml
    
    # Update Defence Strategy safely (scoped under server:)
    awk -v def="${DEFENCE}" '
      /^server:/ { in_server = 1 }
      /^[^ ]/ && !/^server:/ { in_server = 0 }
      in_server && /^  strategy:/ { $0 = "  strategy: " def }
      { print }
    ' config.yaml > config.yaml.tmp && mv config.yaml.tmp config.yaml
    
    # Run the simulation
    poetry run simulation
    
    echo "=================================================="
    echo "Completed simulation for Attack: \"$ATTACK\" | Defence: \"$DEFENCE\""
    echo "=================================================="
    echo ""
  done
done

echo "All simulations completed successfully!"

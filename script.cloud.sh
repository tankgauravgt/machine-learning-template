#!/bin/bash

# Initialize variables to maintain state across the REPL session
MACHINE_ID=""
RUN_ID=""

# Trap SIGINT (Ctrl+C) so it interrupts the current command rather than killing the entire REPL
trap 'echo -e "\nCommand interrupted."' SIGINT

while true; do
    # Clear the terminal screen at the start of every loop iteration
    clear 
    
    echo "========================================="
    echo "          Instance Manager REPL          "
    echo "========================================="
    echo "1. Create instance"
    echo "2. Log into SSH (share port 6006)"
    echo "3. Stream logs"
    echo "4. Destroy instance"
    echo "5. Exit"
    echo "========================================="
    read -p "Select an option (1-5): " OPTION

    case $OPTION in
        1)
            echo "Creating instance..."
            OUTPUT=$(jl run . --gpu H200 --script ./script.local.sh --spot --json --keep --no-follow --yes --setup "export HF_TOKEN=$HF_TOKEN && export ACCELERATE_MIXED_PRECISION=fp8")
                        
            # Extract identifiers
            MACHINE_ID=$(echo "$OUTPUT" | jq -r '.machine_id')
            RUN_ID=$(echo "$OUTPUT" | jq -r '.run_id')

            if [ -n "$MACHINE_ID" ] && [ "$MACHINE_ID" != "null" ]; then
                echo "Successfully created Machine ID: [$MACHINE_ID]"
                echo "Successfully created Run ID: [$RUN_ID]"
            else
                echo "Error: Failed to create the instance or parse the JSON output."
            fi
            ;;
        2)
            if [ -z "$MACHINE_ID" ] || [ "$MACHINE_ID" == "null" ]; then
                echo "Error: No active instance. Please execute Option 1 first."
            else
                echo "Initiating SSH connection with port forwarding (6006:6006)..."
                SSH_CMD=$(jl ssh "$MACHINE_ID" --print-command)
                eval "$SSH_CMD -L localhost:6006:localhost:6006"
                echo "SSH session closed."
            fi
            ;;
        3)
            if [ -z "$RUN_ID" ] || [ "$RUN_ID" == "null" ]; then
                echo "Error: No active run. Please execute Option 1 first."
            else
                echo "Streaming logs for Run ID: [$RUN_ID]..."
                echo "(Press Ctrl+C to stop streaming and return to the menu)"
                jl run logs "$RUN_ID" --follow
            fi
            ;;
        4)
            if [ -z "$MACHINE_ID" ] || [ "$MACHINE_ID" == "null" ]; then
                echo "Error: No active instance to destroy."
            else
                echo "Destroying instance [$MACHINE_ID]..."
                jl destroy "$MACHINE_ID" --yes
                echo "Instance destroyed."
                
                # Clear the state variables
                MACHINE_ID=""
                RUN_ID=""
            fi
            ;;
        5)
            echo "Exiting the manager."
            # Clear screen upon exiting to leave a clean terminal
            clear
            break
            ;;
        *)
            echo "Invalid selection. Please enter a numerical value between 1 and 5."
            ;;
    esac
    
    # Pause to allow the user to read the output before the loop restarts and clears the screen
    if [ "$OPTION" != "5" ]; then
        echo ""
        read -p "Press [Enter] to return to the menu..."
    fi
done
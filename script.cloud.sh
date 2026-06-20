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
    echo "5. Storage"
    echo "6. Exit"
    echo "========================================="
    read -p "Select an option (1-6): " OPTION

    case $OPTION in
        1)
            echo "Creating instance..."
            OUTPUT=$(jl run . --gpu "H200" --region "IN2" --script ./script.local.sh --spot --json --keep --no-follow --yes --setup "export HF_TOKEN=$HF_TOKEN && export ACCELERATE_MIXED_PRECISION=fp16")

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
            while true; do
                clear
                echo "========================================="
                echo "           Storage Manager               "
                echo "========================================="
                echo "1. List storage"
                echo "2. Create storage"
                echo "3. Attach storage"
                echo "4. Detach storage"
                echo "5. Destroy storage"
                echo "6. Back"
                echo "========================================="
                read -p "Select an option (1-6): " STORAGE_OPTION

                case $STORAGE_OPTION in
                    1)
                        echo "Listing storage volumes..."
                        jl storage list
                        ;;
                    2)
                        read -p "Enter storage name: " STORAGE_NAME
                        read -p "Enter size in GB: " STORAGE_SIZE
                        echo "Creating storage [$STORAGE_NAME] (${STORAGE_SIZE}GB)..."
                        jl storage create "$STORAGE_NAME" --size "$STORAGE_SIZE"
                        ;;
                    3)
                        if [ -z "$MACHINE_ID" ] || [ "$MACHINE_ID" == "null" ]; then
                            echo "Error: No active instance. Please create an instance first."
                        else
                            read -p "Enter storage name to attach: " STORAGE_NAME
                            echo "Attaching storage [$STORAGE_NAME] to instance [$MACHINE_ID]..."
                            jl storage attach "$STORAGE_NAME" "$MACHINE_ID"
                        fi
                        ;;
                    4)
                        if [ -z "$MACHINE_ID" ] || [ "$MACHINE_ID" == "null" ]; then
                            echo "Error: No active instance."
                        else
                            read -p "Enter storage name to detach: " STORAGE_NAME
                            echo "Detaching storage [$STORAGE_NAME] from instance [$MACHINE_ID]..."
                            jl storage detach "$STORAGE_NAME" "$MACHINE_ID"
                        fi
                        ;;
                    5)
                        read -p "Enter storage name to destroy: " STORAGE_NAME
                        echo "Destroying storage [$STORAGE_NAME]..."
                        jl storage destroy "$STORAGE_NAME" --yes
                        ;;
                    6)
                        break
                        ;;
                    *)
                        echo "Invalid selection. Please enter a value between 1 and 6."
                        ;;
                esac

                echo ""
                read -p "Press [Enter] to return to the storage menu..."
            done
            ;;
        6)
            echo "Exiting the manager."
            # Clear screen upon exiting to leave a clean terminal
            clear
            break
            ;;
        *)
            echo "Invalid selection. Please enter a numerical value between 1 and 6."
            ;;
    esac
    
    # Pause to allow the user to read the output before the loop restarts and clears the screen
    if [ "$OPTION" != "6" ]; then
        echo ""
        read -p "Press [Enter] to return to the menu..."
    fi
done
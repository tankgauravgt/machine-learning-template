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
            FS_FLAG=""
            read -p "Attach a filesystem? (y/n): " ATTACH_FS
            if [[ "$ATTACH_FS" =~ ^[Yy]$ ]]; then
                echo "Fetching available filesystems..."
                jl filesystem list
                read -p "Enter filesystem ID to attach: " FS_ID
                FS_FLAG="--fs-id $FS_ID"
            fi

            echo "Creating instance..."
            OUTPUT=$(jl run . --gpu "H200" --region "IN2" --script ./script.local.sh --spot --json --keep --no-follow --yes --setup "export HF_TOKEN=$HF_TOKEN && export ACCELERATE_MIXED_PRECISION=fp16" $FS_FLAG)

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
                echo "         Filesystem Manager              "
                echo "========================================="
                echo "1. List filesystems"
                echo "2. Create filesystem"
                echo "3. Edit filesystem (resize)"
                echo "4. Remove filesystem"
                echo "5. Back"
                echo "========================================="
                read -p "Select an option (1-5): " FS_OPTION

                case $FS_OPTION in
                    1)
                        echo "Listing filesystems..."
                        jl filesystem list
                        ;;
                    2)
                        read -p "Enter filesystem name (max 30 chars): " FS_NAME
                        read -p "Enter size in GB (50-2048): " FS_SIZE
                        echo "Creating filesystem [$FS_NAME] (${FS_SIZE}GB) in IN2..."
                        jl filesystem create --name "$FS_NAME" --storage "$FS_SIZE" --yes
                        ;;
                    3)
                        read -p "Enter filesystem ID to resize: " FS_ID
                        read -p "Enter new size in GB (must be larger than current): " FS_SIZE
                        echo "Resizing filesystem [$FS_ID] to ${FS_SIZE}GB..."
                        jl filesystem edit "$FS_ID" --storage "$FS_SIZE" --yes
                        ;;
                    4)
                        read -p "Enter filesystem ID to remove: " FS_ID
                        echo "Removing filesystem [$FS_ID]..."
                        jl filesystem remove "$FS_ID" --yes
                        ;;
                    5)
                        break
                        ;;
                    *)
                        echo "Invalid selection. Please enter a value between 1 and 5."
                        ;;
                esac

                echo ""
                read -p "Press [Enter] to return to the filesystem menu..."
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
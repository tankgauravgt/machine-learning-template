#!/bin/bash

trap '' INT
stty -echoctl 2>/dev/null

show_welcome() {
    clear
    echo ""
    echo ""
    echo "----------------------------------------"
    echo "       Welcome to JarvisXP Terminal     "
    echo "----------------------------------------"
    echo ""
    echo "Commands:"
    echo ""
    echo "1. Manage a filesystem."
    echo "2. Manage a CPU instance."
    echo "3. Manage GPU instances."
    echo "4. Manage GPU VMs."
    echo ""
    echo "Type 'exit' to exit the terminal."
    echo "----------------------------------------"
}

show_fs_menu() {
    clear
    echo ""
    echo "----------------------------------------"
    echo "       Manage a Filesystem              "
    echo "----------------------------------------"
    echo ""
    echo "Commands:"
    echo ""
    echo "1. Create filesystem (IN2)."
    echo "2. List filesystems."
    echo "3. Expand filesystem storage."
    echo "4. Delete filesystem."
    echo ""
    echo "Type 'back' to return to main menu."
    echo "Type 'exit' to exit the terminal."
    echo "----------------------------------------"
}

fs_print_list() {
    local region="$1"
    local raw
    raw=$(jl filesystem list --json 2>/dev/null)
    if ! echo "$raw" | jq -e . >/dev/null 2>&1; then
        echo "Failed to fetch filesystem list."
        return 1
    fi
    echo "$raw" | jq -r --arg reg "$region" '
      if type == "array" then .
      elif (type == "object" and has("filesystems")) then .filesystems
      elif (type == "object" and has("items")) then .items
      else [] end
      | (if ($reg | length) > 0 then map(select(.region == $reg)) else . end)
      | if length == 0 then "No filesystems found."
        else
          (["#","ID","Name","Storage(GB)","Region"] | @tsv),
          (to_entries[] | [.key+1, (.value.fs_id|tostring), (.value.fs_name//"-"), ((.value.storage//"-")|tostring), (.value.region//"-")] | @tsv)
        end
    ' 2>/dev/null | column -t -s $'\t'
}

fs_pick_id() {
    local region="$1"
    echo "" >&2
    echo "Filesystems:" >&2
    fs_print_list "$region" >&2
    echo "" >&2
    printf "Enter # or filesystem ID (blank to cancel): " >&2
    read -r pick
    [[ -z "$pick" ]] && return 1
    local raw chosen
    raw=$(jl filesystem list --json 2>/dev/null)
    if ! echo "$raw" | jq -e . >/dev/null 2>&1; then
        echo "Failed to fetch filesystem list." >&2
        sleep 1
        return 1
    fi
    chosen=$(echo "$raw" | jq -r --arg p "$pick" --arg reg "$region" '
        if type == "array" then .
        elif (type == "object" and has("filesystems")) then .filesystems
        elif (type == "object" and has("items")) then .items
        else [] end
        | (if ($reg | length) > 0 then map(select(.region == $reg)) else . end)
        | to_entries[]
        | select(.key+1 == ($p | try tonumber catch -1) or ((.value.fs_id|tostring) == $p))
        | .value.fs_id
    ' 2>/dev/null)
    if [[ -z "$chosen" ]]; then
        echo "No matching filesystem." >&2
        sleep 1
        return 1
    fi
    echo "$chosen"
}

fs_create() {
    echo ""
    printf "Filesystem name: "
    read -r fs_name
    if [[ -z "$fs_name" ]]; then
        echo "Filesystem name is required."
        sleep 1
        return 1
    fi
    printf "Storage (GB, 50-2048) [50]: "
    read -r fs_storage
    fs_storage="${fs_storage:-50}"
    echo ""
    echo "Creating filesystem '$fs_name' ($fs_storage GB) in IN2..."
    local result
    result=$(jl filesystem create --name "$fs_name" --storage "$fs_storage" --region IN2 --yes --json 2>&1)
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "Failed to create filesystem:"
        echo "$result"
        sleep 2
        return 1
    fi
    local fs_id
    fs_id=$(echo "$result" | jq -r '.fs_id // empty' 2>/dev/null)
    if [[ -z "$fs_id" ]]; then
        echo "Filesystem created, but could not parse ID:"
        echo "$result"
    else
        echo "Filesystem created successfully. ID: $fs_id"
    fi
    sleep 2
}

fs_list() {
    echo ""
    fs_print_list
    echo ""
    echo "Press Enter to continue..."
    read -r
}

fs_expand() {
    local target
    target=$(fs_pick_id) || return 1
    printf "New storage size (GB, must be larger than current): "
    read -r new_storage
    if [[ -z "$new_storage" ]]; then
        echo "New storage size is required."
        sleep 1
        return 1
    fi
    echo ""
    echo "Expanding filesystem $target to $new_storage GB..."
    local result
    result=$(jl filesystem edit "$target" --storage "$new_storage" --yes --json 2>&1)
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "Failed to expand filesystem:"
        echo "$result"
        sleep 2
        return 1
    fi
    echo "Filesystem expanded."
    local new_id
    new_id=$(echo "$result" | jq -r '.fs_id // empty' 2>/dev/null)
    if [[ -n "$new_id" ]]; then
        echo "New filesystem ID (may have changed on edit): $new_id"
    fi
    sleep 2
}

fs_delete() {
    local target
    target=$(fs_pick_id) || return 1
    printf "Are you sure you want to delete filesystem %s? (y/N): " "$target"
    read -r confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        echo "Cancelled."
        sleep 1
        return 1
    fi
    echo ""
    echo "Deleting filesystem $target..."
    local result
    result=$(jl filesystem remove "$target" --yes --json 2>&1)
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "Failed to delete filesystem:"
        echo "$result"
        sleep 2
        return 1
    fi
    echo "Filesystem $target deleted."
    sleep 2
}

manage_filesystem() {
    while true; do
        show_fs_menu
        echo ""
        printf "fs> "
        read -r cmd
        case "$cmd" in
            1)
                fs_create
                ;;
            2)
                fs_list
                ;;
            3)
                fs_expand
                ;;
            4)
                fs_delete
                ;;
            back|b)
                break
                ;;
            exit)
                echo "Goodbye!"
                exit 0
                ;;
            "")
                ;;
            *)
                echo "Unknown command: $cmd"
                sleep 0.5
                ;;
        esac
    done
}

show_cpu_menu() {
    clear
    echo ""
    echo "----------------------------------------"
    echo "       Manage a CPU Instance            "
    echo "----------------------------------------"
    echo ""
    echo "Commands:"
    echo ""
    echo "1. Create CPU VM (IN2)."
    echo "2. List instances."
    echo "3. Show instance details."
    echo "4. Pause instance."
    echo "5. Resume instance."
    echo "6. Destroy instance."
    echo ""
    echo "Type 'back' to return to main menu."
    echo "Type 'exit' to exit the terminal."
    echo "----------------------------------------"
}

cpu_print_list() {
    local raw
    raw=$(jl list --json 2>/dev/null)
    if ! echo "$raw" | jq -e . >/dev/null 2>&1; then
        echo "Failed to fetch instance list."
        return 1
    fi
    echo "$raw" | jq -r '
      if type == "array" then .
      elif (type == "object" and has("instances")) then .instances
      elif (type == "object" and has("items")) then .items
      else [] end
      | if length == 0 then "No instances found."
        else
          (["#","ID","Name","State","Region","vCPUs","RAM(GB)","Storage(GB)"] | @tsv),
          (to_entries[] | [.key+1, (.value.machine_id|tostring), (.value.name//"-"), (.value.state//"-"), (.value.region//"-"), ((.value.vcpus//"-")|tostring), ((.value.ram//"-")|tostring), ((.value.storage//"-")|tostring)] | @tsv)
        end
    ' 2>/dev/null | column -t -s $'\t'
}

cpu_pick_id() {
    echo "" >&2
    echo "Instances:" >&2
    cpu_print_list >&2
    echo "" >&2
    printf "Enter # or instance ID (blank to cancel): " >&2
    read -r pick
    [[ -z "$pick" ]] && return 1
    local raw chosen
    raw=$(jl list --json 2>/dev/null)
    if ! echo "$raw" | jq -e . >/dev/null 2>&1; then
        echo "Failed to fetch instance list." >&2
        sleep 1
        return 1
    fi
    chosen=$(echo "$raw" | jq -r --arg p "$pick" '
        if type == "array" then .
        elif (type == "object" and has("instances")) then .instances
        elif (type == "object" and has("items")) then .items
        else [] end
        | to_entries[]
        | select(.key+1 == ($p | try tonumber catch -1) or ((.value.machine_id|tostring) == $p))
        | .value.machine_id
    ' 2>/dev/null)
    if [[ -z "$chosen" ]]; then
        echo "No matching instance." >&2
        sleep 1
        return 1
    fi
    echo "$chosen"
}

cpu_print_configs() {
    local raw
    raw=$(jl cpus --json 2>/dev/null)
    if ! echo "$raw" | jq -e . >/dev/null 2>&1; then
        echo "Failed to fetch CPU configurations."
        return 1
    fi
    echo "$raw" | jq -r '
      (.combinations // [])
      | map(select(.available and (.regions.IN2 // false)))
      | if length == 0 then "No CPU VM configurations available in IN2."
        else
          (["#","vCPUs","RAM(GB)","Price(\/hr)"] | @tsv),
          (to_entries[] | [.key+1, (.value.vcpus|tostring), (.value.ram_gb|tostring), (.value.price|tostring)] | @tsv)
        end
    ' 2>/dev/null | column -t -s $'\t'
}

cpu_pick_config() {
    echo "" >&2
    echo "Available CPU VM configurations (IN2):" >&2
    cpu_print_configs >&2
    echo "" >&2
    printf "Enter # to select (blank to cancel): " >&2
    read -r pick
    [[ -z "$pick" ]] && return 1
    local raw chosen
    raw=$(jl cpus --json 2>/dev/null)
    if ! echo "$raw" | jq -e . >/dev/null 2>&1; then
        echo "Failed to fetch CPU configurations." >&2
        sleep 1
        return 1
    fi
    chosen=$(echo "$raw" | jq -r --arg p "$pick" '
        (.combinations // [])
        | map(select(.available and (.regions.IN2 // false)))
        | to_entries[]
        | select(.key+1 == ($p | try tonumber catch -1))
        | "\(.value.vcpus) \(.value.ram_gb)"
    ' 2>/dev/null)
    if [[ -z "$chosen" ]]; then
        echo "No matching configuration." >&2
        sleep 1
        return 1
    fi
    echo "$chosen"
}

cpu_create() {
    local config vcpus ram
    config=$(cpu_pick_config) || return 1
    vcpus="${config% *}"
    ram="${config#* }"
    echo "" >&2
    printf "Instance name [cpu-vm]: " >&2
    read -r name
    name="${name:-cpu-vm}"
    printf "Storage in GB [100]: " >&2
    read -r storage
    storage="${storage:-100}"

    echo ""
    echo "Creating CPU VM '$name' ($vcpus vCPUs, $ram GB RAM) in IN2..."
    local result
    result=$(jl create --vm --cpu --name "$name" --storage "$storage" --vcpus "$vcpus" --ram "$ram" --region IN2 --yes --json 2>&1)
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "Failed to create CPU VM:"
        echo "$result"
        sleep 2
        return 1
    fi
    local mid
    mid=$(echo "$result" | jq -r '.machine_id // empty' 2>/dev/null)
    if [[ -z "$mid" ]]; then
        echo "CPU VM may have been created, but could not parse machine_id:"
        echo "$result"
    else
        echo "CPU VM created. Machine ID: $mid"
    fi
    sleep 2
}

cpu_list() {
    echo ""
    cpu_print_list
    echo ""
    echo "Press Enter to continue..."
    read -r
}

cpu_get() {
    local target
    target=$(cpu_pick_id) || return 1
    echo ""
    echo "Details for instance $target:"
    jl get "$target" --json 2>&1 | jq . 2>/dev/null || jl get "$target" --json 2>&1
    echo ""
    echo "Press Enter to continue..."
    read -r
}

cpu_pause() {
    local target
    target=$(cpu_pick_id) || return 1
    printf "Pause instance %s? (y/N): " "$target"
    read -r confirm
    [[ "$confirm" != "y" && "$confirm" != "Y" ]] && { echo "Cancelled."; sleep 1; return 1; }
    echo ""
    echo "Pausing instance $target..."
    local result
    result=$(jl pause "$target" --yes --json 2>&1)
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "Failed to pause instance:"
        echo "$result"
        sleep 2
        return 1
    fi
    echo "Instance $target paused."
    sleep 2
}

cpu_resume() {
    local target
    target=$(cpu_pick_id) || return 1
    printf "Resume instance %s? (y/N): " "$target"
    read -r confirm
    [[ "$confirm" != "y" && "$confirm" != "Y" ]] && { echo "Cancelled."; sleep 1; return 1; }
    echo ""
    echo "Resuming instance $target..."
    local result
    result=$(jl resume "$target" --yes --json 2>&1)
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "Failed to resume instance:"
        echo "$result"
        sleep 2
        return 1
    fi
    local new_id
    new_id=$(echo "$result" | jq -r '.machine_id // empty' 2>/dev/null)
    if [[ -n "$new_id" && "$new_id" != "$target" ]]; then
        echo "Instance resumed. New machine ID: $new_id (ID may change on resume)"
    else
        echo "Instance $target resumed."
    fi
    sleep 2
}

cpu_destroy() {
    local target
    target=$(cpu_pick_id) || return 1
    printf "Permanently destroy instance %s? (y/N): " "$target"
    read -r confirm
    [[ "$confirm" != "y" && "$confirm" != "Y" ]] && { echo "Cancelled."; sleep 1; return 1; }
    echo ""
    echo "Destroying instance $target..."
    local result
    result=$(jl destroy "$target" --yes --json 2>&1)
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "Failed to destroy instance:"
        echo "$result"
        sleep 2
        return 1
    fi
    echo "Instance $target destroyed."
    sleep 2
}

manage_cpu() {
    while true; do
        show_cpu_menu
        echo ""
        printf "cpu> "
        read -r cmd
        case "$cmd" in
            1)
                cpu_create
                ;;
            2)
                cpu_list
                ;;
            3)
                cpu_get
                ;;
            4)
                cpu_pause
                ;;
            5)
                cpu_resume
                ;;
            6)
                cpu_destroy
                ;;
            back|b)
                break
                ;;
            exit)
                echo "Goodbye!"
                exit 0
                ;;
            "")
                ;;
            *)
                echo "Unknown command: $cmd"
                sleep 0.5
                ;;
        esac
    done
}

show_gpu_menu() {
    clear
    echo ""
    echo "----------------------------------------"
    echo "         Manage GPU Instances           "
    echo "----------------------------------------"
    echo ""
    echo "Commands:"
    echo ""
    echo "1. Create GPU instance (IN2)."
    echo "2. List instances."
    echo "3. Show instance details."
    echo "4. Pause instance."
    echo "5. Resume instance."
    echo "6. Destroy instance."
    echo ""
    echo "Type 'back' to return to main menu."
    echo "Type 'exit' to exit the terminal."
    echo "----------------------------------------"
}

gpu_print_configs() {
    local raw
    raw=$(jl gpus --json 2>/dev/null)
    if ! echo "$raw" | jq -e . >/dev/null 2>&1; then
        echo "Failed to fetch GPU list."
        return 1
    fi
    echo "$raw" | jq -r '
      map(select(.region == "IN2" and (.workload_type == "container" or .workload_type == null)))
      | map(select((.effective_num_free_devices // .num_free_devices // 0) > 0))
      | if length == 0 then "No GPU instances available in IN2."
        else
          (["#","GPU","VRAM(GB)","Price(\/hr)","Spot(\/hr)"] | @tsv),
          (to_entries[] | [.key+1, (.value.gpu_type//"-"), (.value.vram//"-"), ((.value.price_per_hour//"-")|tostring), ((.value.spot_price//"-")|tostring)] | @tsv)
        end
    ' 2>/dev/null | column -t -s $'\t'
}

gpu_pick_config() {
    echo "" >&2
    echo "Available GPU instances in IN2:" >&2
    gpu_print_configs >&2
    echo "" >&2
    printf "Enter # to select (blank to cancel): " >&2
    read -r pick
    [[ -z "$pick" ]] && return 1
    local raw chosen
    raw=$(jl gpus --json 2>/dev/null)
    if ! echo "$raw" | jq -e . >/dev/null 2>&1; then
        echo "Failed to fetch GPU list." >&2
        sleep 1
        return 1
    fi
    chosen=$(echo "$raw" | jq -r --arg p "$pick" '
        map(select(.region == "IN2" and (.workload_type == "container" or .workload_type == null)))
        | map(select((.effective_num_free_devices // .num_free_devices // 0) > 0))
        | to_entries[]
        | select(.key+1 == ($p | try tonumber catch -1))
        | .value.gpu_type
    ' 2>/dev/null)
    if [[ -z "$chosen" ]]; then
        echo "No matching GPU." >&2
        sleep 1
        return 1
    fi
    echo "$chosen"
}

gpu_create() {
    local gpu
    gpu=$(gpu_pick_config) || return 1
    echo "" >&2
    printf "Instance name [gpu-container]: " >&2
    read -r name
    name="${name:-gpu-container}"
    printf "Storage in GB [100]: " >&2
    read -r storage
    storage="${storage:-100}"
    printf "Number of GPUs [1]: " >&2
    read -r num_gpus
    num_gpus="${num_gpus:-1}"
    printf "Template [pytorch]: " >&2
    read -r template
    template="${template:-pytorch}"
    printf "Spot instance? (y/N): " >&2
    read -r spot

    local args=(--gpu "$gpu" --storage "$storage" --num-gpus "$num_gpus" --template "$template" --region IN2 --yes --json)
    [[ "$spot" == "y" || "$spot" == "Y" ]] && args+=(--spot)

    echo ""
    echo "Creating GPU instance '$name' ($gpu x$num_gpus, $storage GB) in IN2..."
    local result
    result=$(jl create "${args[@]}" 2>&1)
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "Failed to create GPU instance:"
        echo "$result"
        sleep 2
        return 1
    fi
    local mid
    mid=$(echo "$result" | jq -r '.machine_id // empty' 2>/dev/null)
    if [[ -z "$mid" ]]; then
        echo "GPU instance may have been created, but could not parse machine_id:"
        echo "$result"
    else
        echo "GPU instance created. Machine ID: $mid"
    fi
    sleep 2
}

manage_gpu() {
    while true; do
        show_gpu_menu
        echo ""
        printf "gpu> "
        read -r cmd
        case "$cmd" in
            1)
                gpu_create
                ;;
            2)
                gpu_list
                ;;
            3)
                gpu_get
                ;;
            4)
                gpu_pause
                ;;
            5)
                gpu_resume
                ;;
            6)
                gpu_destroy
                ;;
            back|b)
                break
                ;;
            exit)
                echo "Goodbye!"
                exit 0
                ;;
            "")
                ;;
            *)
                echo "Unknown command: $cmd"
                sleep 0.5
                ;;
        esac
    done
}

show_gpu_vm_menu() {
    clear
    echo ""
    echo "----------------------------------------"
    echo "           Manage GPU VMs               "
    echo "----------------------------------------"
    echo ""
    echo "Commands:"
    echo ""
    echo "1. Create GPU VM (IN2)."
    echo "2. List instances."
    echo "3. Show instance details."
    echo "4. Pause instance."
    echo "5. Resume instance."
    echo "6. Destroy instance."
    echo ""
    echo "Type 'back' to return to main menu."
    echo "Type 'exit' to exit the terminal."
    echo "----------------------------------------"
}

gpu_print_list() {
    local raw
    raw=$(jl list --json 2>/dev/null)
    if ! echo "$raw" | jq -e . >/dev/null 2>&1; then
        echo "Failed to fetch instance list."
        return 1
    fi
    echo "$raw" | jq -r '
      if type == "array" then .
      elif (type == "object" and has("instances")) then .instances
      elif (type == "object" and has("items")) then .items
      else [] end
      | if length == 0 then "No instances found."
        else
          (["#","ID","Name","State","Region","GPU","Storage(GB)"] | @tsv),
          (to_entries[] | [.key+1, (.value.machine_id|tostring), (.value.name//"-"), (.value.state//"-"), (.value.region//"-"), (.value.gpu//"-"), ((.value.storage//"-")|tostring)] | @tsv)
        end
    ' 2>/dev/null | column -t -s $'\t'
}

gpu_pick_id() {
    echo "" >&2
    echo "Instances:" >&2
    gpu_print_list >&2
    echo "" >&2
    printf "Enter # or instance ID (blank to cancel): " >&2
    read -r pick
    [[ -z "$pick" ]] && return 1
    local raw chosen
    raw=$(jl list --json 2>/dev/null)
    if ! echo "$raw" | jq -e . >/dev/null 2>&1; then
        echo "Failed to fetch instance list." >&2
        sleep 1
        return 1
    fi
    chosen=$(echo "$raw" | jq -r --arg p "$pick" '
        if type == "array" then .
        elif (type == "object" and has("instances")) then .instances
        elif (type == "object" and has("items")) then .items
        else [] end
        | to_entries[]
        | select(.key+1 == ($p | try tonumber catch -1) or ((.value.machine_id|tostring) == $p))
        | .value.machine_id
    ' 2>/dev/null)
    if [[ -z "$chosen" ]]; then
        echo "No matching instance." >&2
        sleep 1
        return 1
    fi
    echo "$chosen"
}

gpu_vm_print_configs() {
    local raw
    raw=$(jl gpus --json 2>/dev/null)
    if ! echo "$raw" | jq -e . >/dev/null 2>&1; then
        echo "Failed to fetch GPU list."
        return 1
    fi
    echo "$raw" | jq -r '
      map(select(.region == "IN2" and .workload_type == "vm"))
      | map(select((.effective_num_free_devices // .num_free_devices // 0) > 0))
      | if length == 0 then "No GPU VMs available in IN2."
        else
          (["#","GPU","VRAM(GB)","Price(\/hr)"] | @tsv),
          (to_entries[] | [.key+1, (.value.gpu_type//"-"), (.value.vram//"-"), ((.value.price_per_hour//"-")|tostring)] | @tsv)
        end
    ' 2>/dev/null | column -t -s $'\t'
}

gpu_vm_pick_config() {
    echo "" >&2
    echo "Available GPU VMs in IN2:" >&2
    gpu_vm_print_configs >&2
    echo "" >&2
    printf "Enter # to select (blank to cancel): " >&2
    read -r pick
    [[ -z "$pick" ]] && return 1
    local raw chosen
    raw=$(jl gpus --json 2>/dev/null)
    if ! echo "$raw" | jq -e . >/dev/null 2>&1; then
        echo "Failed to fetch GPU list." >&2
        sleep 1
        return 1
    fi
    chosen=$(echo "$raw" | jq -r --arg p "$pick" '
        map(select(.region == "IN2" and .workload_type == "vm"))
        | map(select((.effective_num_free_devices // .num_free_devices // 0) > 0))
        | to_entries[]
        | select(.key+1 == ($p | try tonumber catch -1))
        | .value.gpu_type
    ' 2>/dev/null)
    if [[ -z "$chosen" ]]; then
        echo "No matching GPU." >&2
        sleep 1
        return 1
    fi
    echo "$chosen"
}

gpu_vm_create() {
    local gpu
    gpu=$(gpu_vm_pick_config) || return 1
    echo "" >&2
    printf "Instance name [gpu-vm]: " >&2
    read -r name
    name="${name:-gpu-vm}"
    printf "Storage in GB [100]: " >&2
    read -r storage
    storage="${storage:-100}"
    printf "Number of GPUs [1]: " >&2
    read -r num_gpus
    num_gpus="${num_gpus:-1}"

    local args=(--vm --gpu "$gpu" --storage "$storage" --num-gpus "$num_gpus" --region IN2 --yes --json)

    echo ""
    echo "Creating GPU VM '$name' ($gpu x$num_gpus, $storage GB) in IN2..."
    local result
    result=$(jl create "${args[@]}" 2>&1)
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "Failed to create GPU VM:"
        echo "$result"
        sleep 2
        return 1
    fi
    local mid
    mid=$(echo "$result" | jq -r '.machine_id // empty' 2>/dev/null)
    if [[ -z "$mid" ]]; then
        echo "GPU VM may have been created, but could not parse machine_id:"
        echo "$result"
    else
        echo "GPU VM created. Machine ID: $mid"
    fi
    sleep 2
}

gpu_list() {
    echo ""
    gpu_print_list
    echo ""
    echo "Press Enter to continue..."
    read -r
}

gpu_get() {
    local target
    target=$(gpu_pick_id) || return 1
    echo ""
    echo "Details for instance $target:"
    jl get "$target" --json 2>&1 | jq . 2>/dev/null || jl get "$target" --json 2>&1
    echo ""
    echo "Press Enter to continue..."
    read -r
}

gpu_pause() {
    local target
    target=$(gpu_pick_id) || return 1
    printf "Pause instance %s? (y/N): " "$target"
    read -r confirm
    [[ "$confirm" != "y" && "$confirm" != "Y" ]] && { echo "Cancelled."; sleep 1; return 1; }
    echo ""
    echo "Pausing instance $target..."
    local result
    result=$(jl pause "$target" --yes --json 2>&1)
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "Failed to pause instance:"
        echo "$result"
        sleep 2
        return 1
    fi
    echo "Instance $target paused."
    sleep 2
}

gpu_resume() {
    local target
    target=$(gpu_pick_id) || return 1
    printf "Resume instance %s? (y/N): " "$target"
    read -r confirm
    [[ "$confirm" != "y" && "$confirm" != "Y" ]] && { echo "Cancelled."; sleep 1; return 1; }
    echo ""
    echo "Resuming instance $target..."
    local result
    result=$(jl resume "$target" --yes --json 2>&1)
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "Failed to resume instance:"
        echo "$result"
        sleep 2
        return 1
    fi
    local new_id
    new_id=$(echo "$result" | jq -r '.machine_id // empty' 2>/dev/null)
    if [[ -n "$new_id" && "$new_id" != "$target" ]]; then
        echo "Instance resumed. New machine ID: $new_id (ID may change on resume)"
    else
        echo "Instance $target resumed."
    fi
    sleep 2
}

gpu_destroy() {
    local target
    target=$(gpu_pick_id) || return 1
    printf "Permanently destroy instance %s? (y/N): " "$target"
    read -r confirm
    [[ "$confirm" != "y" && "$confirm" != "Y" ]] && { echo "Cancelled."; sleep 1; return 1; }
    echo ""
    echo "Destroying instance $target..."
    local result
    result=$(jl destroy "$target" --yes --json 2>&1)
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "Failed to destroy instance:"
        echo "$result"
        sleep 2
        return 1
    fi
    echo "Instance $target destroyed."
    sleep 2
}

manage_gpu_vm() {
    while true; do
        show_gpu_vm_menu
        echo ""
        printf "gpu> "
        read -r cmd
        case "$cmd" in
            1)
                gpu_vm_create
                ;;
            2)
                gpu_list
                ;;
            3)
                gpu_get
                ;;
            4)
                gpu_pause
                ;;
            5)
                gpu_resume
                ;;
            6)
                gpu_destroy
                ;;
            back|b)
                break
                ;;
            exit)
                echo "Goodbye!"
                exit 0
                ;;
            "")
                ;;
            *)
                echo "Unknown command: $cmd"
                sleep 0.5
                ;;
        esac
    done
}

main() {
    while true; do
        show_welcome
        echo ""
        printf "> "
        read -r cmd
        case "$cmd" in
            exit)
                echo "Goodbye!"
                break
                ;;
            1)
                manage_filesystem
                ;;
            2)
                manage_cpu
                ;;
            3)
                manage_gpu
                ;;
            4)
                manage_gpu_vm
                ;;
            "")
                ;;
            *)
                echo "Unknown command: $cmd"
                sleep 0.5
                ;;
        esac
    done
}

main

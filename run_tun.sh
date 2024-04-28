#!/bin/sh

# Set local ports
LOCAL_PORT_1=3745
LOCAL_PORT_2=6006

# Define tunnel names
TUNNEL_NAME_1="ax"
TUNNEL_NAME_2="tensorboard"

# Check if ngrok.yml exists and create it if not
NGROK_CONFIG_FILE="/root/.config/ngrok/ngrok.yml"
if [ ! -f "$NGROK_CONFIG_FILE" ]; then
    mkdir -p "$(dirname "$NGROK_CONFIG_FILE")"
    touch "$NGROK_CONFIG_FILE"
fi

# Function to check and add a tunnel if it doesn't exist
add_tunnel_if_not_exists() {
  local name="$1"
  local proto="$2"
  local addr="$3"
  if ! grep -q "^  $name:" "$NGROK_CONFIG_FILE"; then
    echo "  $name:" >> "$NGROK_CONFIG_FILE"
    echo "    proto: $proto" >> "$NGROK_CONFIG_FILE"
    echo "    addr: $addr" >> "$NGROK_CONFIG_FILE"
  fi
}

# Find the line number where we want to insert "tunnels:"
insert_line_num=$(grep -n "^authtoken:" "$NGROK_CONFIG_FILE" | cut -d: -f1)
# Increment the line number to insert after the "authtoken" line
insert_line_num=$((insert_line_num + 1))

# Check if "tunnels:" already exists
if ! grep -q "^tunnels:" "$NGROK_CONFIG_FILE"; then
  # Insert "tunnels:" at the calculated line number
  sed -i "${insert_line_num}i tunnels:" "$NGROK_CONFIG_FILE"
fi

# Add tunnels only if they don't already exist
add_tunnel_if_not_exists "$TUNNEL_NAME_1" "http" "$LOCAL_PORT_1"
add_tunnel_if_not_exists "$TUNNEL_NAME_2" "http" "$LOCAL_PORT_2"

echo "Start ngrok in background with all tunnels"
ngrok start --all >/dev/null &

echo "Waiting for 5 seconds for tunnels to initialize..."
sleep 5 

echo "Extracting ngrok public URLs..."

# Get the tunnel information for each tunnel name
get_tunnel_info() {
  local tunnel_name="$1"
  curl --silent http://127.0.0.1:4040/api/tunnels | jq -r '.tunnels[] | select(.name=="'$tunnel_name'") | .public_url'
}

NGROK_PUBLIC_URL_1=$(get_tunnel_info "$TUNNEL_NAME_1")
NGROK_PUBLIC_URL_2=$(get_tunnel_info "$TUNNEL_NAME_2")

echo ""
echo "AX RVC => [ $NGROK_PUBLIC_URL_1 ]"
echo "Tensonboard => [ $NGROK_PUBLIC_URL_2 ]"
echo ""

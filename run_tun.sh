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
curl http://127.0.0.1:4040/api/tunnels 

echo -n "Extracting ngrok public URLs ."
NGROK_PUBLIC_URL_1=""
NGROK_PUBLIC_URL_2=""
while [ -z "$NGROK_PUBLIC_URL_1" ] || [ -z "$NGROK_PUBLIC_URL_2" ]; do
  # Run 'curl' against ngrok API and extract public URLs
  RESPONSE=$(curl --silent --max-time 10 --connect-timeout 5 \
                  --show-error http://127.0.0.1:4040/api/tunnels)
  # Extract URLs using tunnel names in sed
  NGROK_PUBLIC_URL_1=$(echo "$RESPONSE" | sed -nE 's/.*"'"$TUNNEL_NAME_1"'"":{"public_url":"https:..([^"]*).*/\1/p')
  NGROK_PUBLIC_URL_2=$(echo "$RESPONSE" | sed -nE 's/.*"'"$TUNNEL_NAME_2"'"":{"public_url":"https:..([^"]*).*/\1/p')
  sleep 1
  echo -n "."
done

echo
echo "NGROK_PUBLIC_URL_1 => [ $NGROK_PUBLIC_URL_1 ]"
echo "NGROK_PUBLIC_URL_2 => [ $NGROK_PUBLIC_URL_2 ]"

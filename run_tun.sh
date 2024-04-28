#!/bin/sh

# Set local ports
LOCAL_PORT_1=3745
LOCAL_PORT_2=6006

# Define tunnel names
TUNNEL_NAME_1="ax"
TUNNEL_NAME_2="tensorboard"

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

# Process the ngrok.yml file with awk
awk '
  /^tunnels:/ { found_tunnels=1; print; next }  # Print existing "tunnels:" line
  found_tunnels { print }                       # Print lines after "tunnels:"
  { buffer=(buffer ? buffer ORS : "") $0 }      # Store other lines in buffer
  END { 
    if (!found_tunnels) {                       # If "tunnels:" not found
      print buffer ORS "tunnels:"               # Print buffered lines and add "tunnels:"
    }
  }
' "$NGROK_CONFIG_FILE" > "$NGROK_CONFIG_FILE.tmp" && mv "$NGROK_CONFIG_FILE.tmp" "$NGROK_CONFIG_FILE"

# Add tunnels only if they don't already exist
add_tunnel_if_not_exists "$TUNNEL_NAME_1" "http" "$LOCAL_PORT_1"
add_tunnel_if_not_exists "$TUNNEL_NAME_2" "http" "$LOCAL_PORT_2"

echo "Start ngrok in background with all tunnels"
nohup ngrok start --all &>/dev/null &

echo -n "Extracting ngrok public URLs ."
NGROK_PUBLIC_URL_1=""
NGROK_PUBLIC_URL_2=""
while [ -z "$NGROK_PUBLIC_URL_1" ] || [ -z "$NGROK_PUBLIC_URL_2" ]; do
  # Run 'curl' against ngrok API and extract public URLs (using 'sed' command)
  RESPONSE=$(curl --silent --max-time 10 --connect-timeout 5 \
                  --show-error http://127.0.0.1:4040/api/tunnels)
  NGROK_PUBLIC_URL_1=$(echo "$RESPONSE" | sed -nE 's/.*public_url":"https:..([^"]*).*/\1/p')
  NGROK_PUBLIC_URL_2=$(echo "$RESPONSE" | sed -nE 's/.*public_url":"https:..([^"]*).*/\1/p')
  sleep 1
  echo -n "."
done

echo
echo "NGROK_PUBLIC_URL_1 => [ $NGROK_PUBLIC_URL_1 ]"
echo "NGROK_PUBLIC_URL_2 => [ $NGROK_PUBLIC_URL_2 ]"

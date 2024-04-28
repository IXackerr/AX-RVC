#!/bin/sh

# Set local ports
LOCAL_PORT_1=3745
LOCAL_PORT_2=6006

# Check if ngrok.yml exists and create it if not
NGROK_CONFIG_FILE="/root/.config/ngrok/ngrok.yml"
if [ ! -f "$NGROK_CONFIG_FILE" ]; then
    mkdir -p "$(dirname "$NGROK_CONFIG_FILE")"
    touch "$NGROK_CONFIG_FILE"
fi

# Check if the specified tunnels exist in ngrok.yml
TUNNELS_EXIST=$(grep -c "tunnels:\s*\ntensorboard:\s*\n\s*proto: http\s*\n\s*addr: $LOCAL_PORT_2\s*\nax:\s*\n\s*proto: http\s*\n\s*addr: $LOCAL_PORT_1" "$NGROK_CONFIG_FILE")

# Add missing tunnels configuration to ngrok.yml
if [ "$TUNNELS_EXIST" -eq 0 ]; then
    echo "tunnels:" >> "$NGROK_CONFIG_FILE"
    echo "  tensorboard:" >> "$NGROK_CONFIG_FILE"
    echo "    proto: http" >> "$NGROK_CONFIG_FILE"
    echo "    addr: $LOCAL_PORT_2" >> "$NGROK_CONFIG_FILE"
    echo "  ax:" >> "$NGROK_CONFIG_FILE"
    echo "    proto: http" >> "$NGROK_CONFIG_FILE"
    echo "    addr: $LOCAL_PORT_1" >> "$NGROK_CONFIG_FILE"
fi

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

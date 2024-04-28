#!/bin/sh

# Set local ports
LOCAL_PORT_1=3745
LOCAL_PORT_2=6006

echo "Start ngrok in background on ports [ $LOCAL_PORT_1 ] and [ $LOCAL_PORT_2 ]"
nohup ngrok http $LOCAL_PORT_1 &>/dev/null &
nohup ngrok http $LOCAL_PORT_2 &>/dev/null &

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

function gdrive_download() {
  CONFIRM=$(curl -sc /tmp/gcookie "https://drive.google.com/uc?export=download&id=$1" | \
            grep -o 'confirm=[^&]*' | sed 's/confirm=//')
  curl -Lb /tmp/gcookie "https://drive.google.com/uc?export=download&confirm=$CONFIRM&id=$1" \
       -o $2
}

# Usage
gdrive_download 1TEOR60JXgOkPrC6jSLhuR2Nb6eCegjpd asr_data.tar
# Create the data directory if it doesn't exist
mkdir -p ./data

# Declare an associative array to hold the URLs and their corresponding filenames
declare -A urls=(
    ["v1-5-pruned-emaonly.ckpt"]="https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt"
    ["merges.txt"]="https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/merges.txt"
    ["vocab.json"]="https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/tokenizer/vocab.json"
)

# Loop through the array and download each file
for filename in "${!urls[@]}"; do
    url="${urls[$filename]}"
    echo "Downloading $filename from $url..."
    
    # Use curl on macOS or wget on Linux to download the file
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        curl -o "./data/$filename" "$url"
    else
        # Linux
        wget -O "./data/$filename" "$url"
    fi
done

echo "Download complete!"

#!/bin/bash

# Set Streamlit configuration for large uploads
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1024
export STREAMLIT_SERVER_MAX_MESSAGE_SIZE=1024

# Create .streamlit directory if it doesn't exist
mkdir -p ~/.streamlit

# Create config file with large upload settings
cat > ~/.streamlit/config.toml << EOF
[server]
maxUploadSize = 1024
maxMessageSize = 1024
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
EOF

# Run the Streamlit app
streamlit run streamlit_app.py --server.port=7860 --server.address=0.0.0.0 --server.maxUploadSize=1024

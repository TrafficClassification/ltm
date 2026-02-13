export TOKENIZERS_PARALLELISM=false
PROMPT_ustc_tfc2016_malware="Classify this network traffic data into one application category: Htbot, Geodo, Virut, Neris, Shifu, Cridex, Tinba, Zeus, Miuref, Nsis. Output the category name ONLY. Do NOT add any punctuation, explanation, or extra text."
PROMPT_ustc_tfc2016_benign="Classify this network traffic data into one application category: BitTorrent, FaceTime, FTP, Gmail, MySQL, Outlook, Skype, SMB, Weibo, WorldOfWarcraft. Output the category name ONLY. Do NOT add any punctuation, explanation, or extra text."
PROMPT_iscx_tor="Classify this network traffic data into one application category: Browsing,Email, Chat, Audio-Streaming, Video-Streaming, FTP, VoIP, P2P. Output the category name ONLY. Do NOT add any punctuation, explanation, or extra text."
PROMPT_iscx_vpn="Classify this network traffic data into one application category: Email, FileTransfer, TraP2P, VoIP, Chat, Streaming. Output the category name ONLY. Do NOT add any punctuation, explanation, or extra text."
python preprocess_dataset.py \
    --input /home/xjtu/workspace/dataset/ustc_tfc2016/Benign_balanced_6000 \
    --output_path /home/xjtu/workspace/ltm/dataset/first_n/pac5 \
    --output_name ustc_benign_1token \
    --num_workers 12 \
    --system_prompt "$PROMPT_ustc_tfc2016_benign"

export TOKENIZERS_PARALLELISM=false
PROMPT_ustc_tfc2016_malware="Classify this network traffic data into one application category: Htbot, Geodo, Virut, Neris, Shifu, Cridex, Tinba, Zeus, Miuref, Nsis. Output the category name ONLY. Do NOT add any punctuation, explanation, or extra text."
PROMPT_ustc_tfc2016_benign="Classify this network traffic data into one application category: BitTorrent, FaceTime, FTP, Gmail, MySQL, Outlook, Skype, SMB, Weibo, WorldOfWarcraft. Output the category name ONLY. Do NOT add any punctuation, explanation, or extra text."
PROMPT_iscx_tor="Classify this network traffic data into one application category: Browsing,Email, Chat, Audio-Streaming, Video-Streaming, FTP, VoIP, P2P. Output the category name ONLY. Do NOT add any punctuation, explanation, or extra text."
PROMPT_iscx_vpn="Classify this network traffic data into one application category: Email, FileTransfer, TraP2P, VoIP, Chat, Streaming. Output the category name ONLY. Do NOT add any punctuation, explanation, or extra text."
python wo_osp.py \
    --input /home/xjtu/workspace/dataset/iscxvpn_nonvpn/vpn_services \
    --output_path /home/xjtu/workspace/ltm/dataset/ablation/vpn_services \
    --output_name vpn_services_wo_osp \
    --num_workers 16 \
    --system_prompt "$PROMPT_iscx_vpn"

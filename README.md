# 手順

1. speckle_generate.py を利用して画像サイズ分の time_length のスペックルを作成
2. data_process.py の divide_mask を利用して 75%, 50%, 25%にそれぞれスペックルを分割して保存
3. train.py の size, speckle_num を変更して学習させる。

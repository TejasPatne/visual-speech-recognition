#preparing for download 
mkdir "gridcorpus"
cd "gridcorpus"
mkdir "raw" "align" "video"
cd "raw" && mkdir "align" "video"

for ((i=1; i<=34; i++));
do
  if [[ $i -ne 21 ]]; then
    printf "\n\n------------------------- Downloading $i th speaker -------------------------\n\n"
    
    #download the audio of the ith speaker
    cd "align" && curl "http://spandh.dcs.shef.ac.uk/gridcorpus/s$i/align/s$i.tar" > "s$i.tar" && cd ..
    cd "video" && curl "http://spandh.dcs.shef.ac.uk/gridcorpus/s$i/video/s$i.mpg_vcd.zip" > "s$i.zip" && cd ..

    read -p "Extract downloaded files? (y/N): " answer
    if [[ $answer =~ ^[Yy]$ ]];
    then
        unzip -q "./video/s$i.zip" -d "../video"
        tar -xf "./align/s$i.tar" -C "../align"
    fi
  fi
done
rename 's/\d+/sprintf("%05d",$&)/e' *.png
ffmpeg -i %05d.png -vf palettegen palette.png
ffmpeg -framerate 60 -i %05d.png -i palette.png -lavfi "paletteuse,setpts=6*PTS" -y play.gif

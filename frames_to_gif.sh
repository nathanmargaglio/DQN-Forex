cd dqn/frames
#rename 's/\d+/sprintf("%05d",$&)/e' *.png
ffmpeg -i %05d.png -vf palettegen palette.png -y
ffmpeg -framerate 60 -i %05d.png -i palette.png -lavfi "paletteuse,setpts=6*PTS" -y play.gif
ffmpeg -framerate 120 -i %05d.png -i palette.png -lavfi "paletteuse,setpts=6*PTS" -y playx2.gif
mv play.gif ../play.gif
mv playx2.gif ../playx2.gif

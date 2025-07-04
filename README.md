

```shell
> ffmpeg -i input-moving.gif -c:v libx264 -pix_fmt yuv420p -profile:v main -level 3.1 -movflags faststart -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" output.mp4
```
